import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from kubernetes import client
from kubernetes.client import ApiException
from typer.testing import CliRunner

from robusta_krr.core.integrations import kubernetes
from robusta_krr.core.integrations.prometheus.metrics_service.prometheus_metrics_service import PrometheusMetricsService
from robusta_krr.core.models.config import Config
from robusta_krr.core.models.objects import PodData
from robusta_krr.main import app, load_commands


@pytest.fixture
def cnpg_resource():
    return {
        "apiVersion": "postgresql.cnpg.io/v1",
        "kind": "Cluster",
        "metadata": {
            "name": "postgres-db",
            "namespace": "databases",
            "labels": {"team": "infra"},
            "annotations": {"description": "database"},
        },
        "spec": {
            "instances": 3,
            "resources": {
                "requests": {"cpu": "500m", "memory": "1Gi"},
                "limits": {"cpu": "2", "memory": "2Gi"},
            },
        },
    }


@pytest.fixture
def cnpg_loader(monkeypatch, cnpg_resource):
    config = Config(
        strategy="simple",
        resources=["cnpgcluster"],
        namespaces=[],
        max_workers=2,
        format="json",
        show_cluster_name=False,
        log_to_stderr=False,
        other_args={},
    )
    monkeypatch.setattr(kubernetes, "settings", config)
    with client.ApiClient() as api_client:
        monkeypatch.setattr(Config, "get_kube_client", lambda self, context=None: api_client)
        loader = kubernetes.ClusterLoader(cluster="test-cluster")
        loader.custom_objects = MagicMock(spec=client.CustomObjectsApi)
        response = {"apiVersion": "postgresql.cnpg.io/v1", "kind": "ClusterList", "items": [cnpg_resource]}
        loader.custom_objects.list_cluster_custom_object.return_value = response
        loader.custom_objects.list_namespaced_custom_object.return_value = response
        loader.core = MagicMock(spec=client.CoreV1Api)
        loader._try_list_hpa = AsyncMock(return_value={})
        try:
            yield loader
        finally:
            loader.executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("namespaces", ["*", ["databases"]])
async def test_cnpg_discovery(cnpg_loader, namespaces):
    kubernetes.settings.namespaces = namespaces
    kubernetes.settings.selector = "team=infra"

    objects = await cnpg_loader.list_scannable_objects()

    assert len(objects) == 1
    obj = objects[0]
    assert (obj.cluster, obj.namespace, obj.name, obj.kind, obj.container) == (
        "test-cluster",
        "databases",
        "postgres-db",
        "CNPGCluster",
        "postgres",
    )
    assert obj.allocations.requests == {"cpu": 0.5, "memory": 1024**3}
    assert obj.allocations.limits == {"cpu": 2, "memory": 2 * 1024**3}
    assert obj.labels == {"team": "infra"}
    assert obj.annotations == {"description": "database"}
    expected = dict(
        group="postgresql.cnpg.io", version="v1", plural="clusters", watch=False, label_selector="team=infra"
    )
    if namespaces == "*":
        cnpg_loader.custom_objects.list_cluster_custom_object.assert_called_once_with(**expected)
        cnpg_loader.custom_objects.list_namespaced_custom_object.assert_not_called()
    else:
        cnpg_loader.custom_objects.list_namespaced_custom_object.assert_called_once_with(
            namespace="databases", **expected
        )
        cnpg_loader.custom_objects.list_cluster_custom_object.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("resources", [None, {}, {"requests": None, "limits": None}, {"requests": {"cpu": "250m"}}])
async def test_cnpg_missing_resource_values(cnpg_loader, cnpg_resource, resources):
    if resources is None:
        cnpg_resource["spec"].pop("resources")
    else:
        cnpg_resource["spec"]["resources"] = resources
    objects = await cnpg_loader.list_scannable_objects()

    assert len(objects) == 1
    assert objects[0].allocations.requests == {
        "cpu": 0.25 if resources and resources.get("requests") else None,
        "memory": None,
    }
    assert objects[0].allocations.limits == {"cpu": None, "memory": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("resources, expected_count", [("*", 1), (["CNPGCluster"], 1), (["Deployment"], 0)])
async def test_cnpg_resource_filter(cnpg_loader, resources, expected_count):
    kubernetes.settings.resources = resources
    cnpg_loader._ClusterLoader__hpa_list = {}

    assert len(await cnpg_loader._list_cnpg_clusters()) == expected_count
    assert cnpg_loader.custom_objects.list_cluster_custom_object.call_count == expected_count


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_cnpg_unavailable_api_is_skipped(cnpg_loader, status):
    cnpg_loader.custom_objects.list_cluster_custom_object.side_effect = ApiException(status=status)

    assert await cnpg_loader.list_scannable_objects() == []
    assert await cnpg_loader.list_scannable_objects() == []
    cnpg_loader.custom_objects.list_cluster_custom_object.assert_called_once()


@pytest.mark.asyncio
async def test_cnpg_kubernetes_pod_fallback(cnpg_loader):
    obj = (await cnpg_loader.list_scannable_objects())[0]
    cnpg_loader.core.list_namespaced_pod.return_value = client.V1PodList(
        items=[client.V1Pod(metadata=client.V1ObjectMeta(name="postgres-db-1"))]
    )

    assert await cnpg_loader.list_pods(obj) == [PodData(name="postgres-db-1", deleted=False)]
    cnpg_loader.core.list_namespaced_pod.assert_called_once_with(
        namespace="databases", label_selector="cnpg.io/cluster=postgres-db"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, owner_kind", [("CNPGCluster", "Cluster"), ("StatefulSet", "StatefulSet")])
async def test_prometheus_pod_owner_kind(cnpg_loader, monkeypatch, kind, owner_kind):
    obj = (await cnpg_loader.list_scannable_objects())[0]
    obj.kind = kind
    # Bypass connection setup; exercise the real pod discovery with mocked query responses.
    service = PrometheusMetricsService.__new__(PrometheusMetricsService)
    monkeypatch.setattr(service, "get_prometheus_cluster_label", lambda: ', cluster="test-cluster"')
    service.query = AsyncMock(
        side_effect=[
            [{"metric": {"pod": "postgres-db-1"}}, {"metric": {"pod": "postgres-db-old"}}],
            [{"metric": {"pod": "postgres-db-1"}}],
        ]
    )

    pods = await service.load_pods(obj, timedelta(days=7))

    owner_query = service.query.await_args_list[0].args[0]
    assert f'owner_kind="{owner_kind}"' in owner_query
    assert 'owner_name=~"postgres-db"' in owner_query
    assert 'namespace="databases"' in owner_query
    assert 'cluster="test-cluster"' in owner_query
    assert "[7d]" in owner_query
    assert set(pods) == {PodData(name="postgres-db-1", deleted=False), PodData(name="postgres-db-old", deleted=True)}


@pytest.mark.parametrize("resources", [None, {"requests": {"cpu": "500m", "memory": "1Gi"}}])
@pytest.mark.parametrize("output_format", ["json", "yaml", "table", "csv", "csv-raw"])
def test_cnpg_cli_output(cnpg_loader, cnpg_resource, monkeypatch, resources, output_format):
    cnpg_resource["spec"]["resources"] = resources
    objects = asyncio.run(cnpg_loader.list_scannable_objects())
    monkeypatch.setattr(kubernetes.KubernetesLoader, "list_scannable_objects", AsyncMock(return_value=objects))
    load_commands()
    result = CliRunner(mix_stderr=False).invoke(
        app,
        ["simple", "--resource", "CNPGCluster", "-n", "databases", "-q", "--width", "240", "-f", output_format],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert "CNPGCluster" in result.stdout
    assert "postgres-db" in result.stdout
    assert "postgres" in result.stdout
