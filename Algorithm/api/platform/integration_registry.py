from __future__ import annotations

from typing import Dict, List

from api.platform.contracts import ConnectorCapability, IntegrationDefinition


_DEFINITIONS: Dict[str, IntegrationDefinition] = {
    "crm": IntegrationDefinition(
        integration_type="crm",
        title="CRM",
        description="Future client, deal, and relationship context connector.",
        capabilities=[
            ConnectorCapability(capability_id="case_linking", title="Case Linking", description="Attach platform dossiers to CRM records."),
        ],
        config_schema={"required": ["endpoint_label"]},
    ),
    "document_storage": IntegrationDefinition(
        integration_type="document_storage",
        title="Document Storage",
        description="Future export archive and memo distribution connector.",
        capabilities=[
            ConnectorCapability(capability_id="pack_archive", title="Pack Archive", description="Store export-grade document packs."),
        ],
        config_schema={"required": ["storage_label"]},
    ),
    "market_data_extension": IntegrationDefinition(
        integration_type="market_data_extension",
        title="Market Data Extension",
        description="Future extension point for enterprise-grade market or fundamental feeds.",
        capabilities=[
            ConnectorCapability(capability_id="provider_extension", title="Provider Extension", description="Extend the current data-fabric source stack."),
        ],
        config_schema={"required": ["provider_name"]},
    ),
    "messaging_notification": IntegrationDefinition(
        integration_type="messaging_notification",
        title="Messaging / Notification",
        description="Future alert and notification connector.",
        capabilities=[
            ConnectorCapability(capability_id="workflow_alerts", title="Workflow Alerts", description="Publish workflow and approval alerts."),
        ],
        config_schema={"required": ["channel"]},
    ),
    "ticketing_tasking": IntegrationDefinition(
        integration_type="ticketing_tasking",
        title="Ticketing / Tasking",
        description="Future tasking and request-management connector.",
        capabilities=[
            ConnectorCapability(capability_id="task_sync", title="Task Sync", description="Mirror workflow actions into enterprise task systems."),
        ],
        config_schema={"required": ["queue_name"]},
    ),
    "portfolio_oms_pms": IntegrationDefinition(
        integration_type="portfolio_oms_pms",
        title="Portfolio / OMS / PMS",
        description="Future portfolio-system connector for governed deployment.",
        capabilities=[
            ConnectorCapability(capability_id="position_sync", title="Position Sync", description="Attach governed AXIOM outputs to portfolio systems."),
        ],
        config_schema={"required": ["system_label"]},
    ),
    "internal_research_archive": IntegrationDefinition(
        integration_type="internal_research_archive",
        title="Internal Research Archive",
        description="Future internal archive for dossiers, exports, and evidence trails.",
        capabilities=[
            ConnectorCapability(capability_id="archive_sync", title="Archive Sync", description="Persist dossier versions and exports to research archives."),
        ],
        config_schema={"required": ["archive_label"]},
    ),
}


def list_integration_definitions() -> List[IntegrationDefinition]:
    return [item.model_copy(deep=True) for item in _DEFINITIONS.values()]


def get_integration_definition(integration_type: str) -> IntegrationDefinition:
    definition = _DEFINITIONS.get(str(integration_type or ""))
    if definition is None:
        return IntegrationDefinition(
            integration_type=str(integration_type or "custom"),
            title=str(integration_type or "Custom").replace("_", " ").title(),
            description="Custom integration binding.",
            capabilities=[],
            config_schema={},
        )
    return definition.model_copy(deep=True)
