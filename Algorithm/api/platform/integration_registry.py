from __future__ import annotations

from typing import Dict, List

from api.platform.contracts import ConnectorCapability, IntegrationDefinition


_DEFINITIONS: Dict[str, IntegrationDefinition] = {
    "local_archive": IntegrationDefinition(
        integration_type="local_archive",
        title="Local Archive",
        description="Pilot-safe local archive for rendered exports and dossier packs.",
        capabilities=[
            ConnectorCapability(
                capability_id="archive_export",
                title="Archive Export",
                description="Persist rendered export packs to a managed local archive path.",
            ),
        ],
        config_schema={"required": ["target_root"]},
    ),
    "webhook": IntegrationDefinition(
        integration_type="webhook",
        title="Webhook Outbox",
        description="Pilot-safe webhook-style outbox for workflow, export, and approval events.",
        capabilities=[
            ConnectorCapability(
                capability_id="event_delivery",
                title="Event Delivery",
                description="Queue structured platform events into a deterministic webhook outbox.",
            ),
        ],
        config_schema={"required": ["outbox_root"]},
    ),
    "internal_sink": IntegrationDefinition(
        integration_type="internal_sink",
        title="Internal Sink",
        description="Internal event sink for audit-compatible platform activity capture.",
        capabilities=[
            ConnectorCapability(
                capability_id="sink_event",
                title="Sink Event",
                description="Write platform execution events into an internal structured sink.",
            ),
        ],
        config_schema={"required": ["sink_path"]},
    ),
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
    "premium_market_data": IntegrationDefinition(
        integration_type="premium_market_data",
        title="Premium Market Data",
        description="Credential-aware premium market-data probe for institutional-grade bar and reference-feed readiness.",
        capabilities=[
            ConnectorCapability(
                capability_id="probe_market_data",
                title="Probe Market Data",
                description="Check premium market-data readiness and optionally run a live probe against the configured feed.",
            ),
        ],
        config_schema={"optional": ["sample_symbol", "execute_live"]},
    ),
    "premium_news_intel": IntegrationDefinition(
        integration_type="premium_news_intel",
        title="Premium News Intelligence",
        description="Credential-aware premium news and headline-intelligence probe.",
        capabilities=[
            ConnectorCapability(
                capability_id="probe_news_intel",
                title="Probe News Intelligence",
                description="Check premium news-intelligence readiness and optionally run a live probe against the configured feed.",
            ),
        ],
        config_schema={"optional": ["sample_symbol", "execute_live"]},
    ),
    "filings_intel": IntegrationDefinition(
        integration_type="filings_intel",
        title="Filings Intelligence",
        description="Credential-aware SEC and filings-intelligence probe for event and filing readiness.",
        capabilities=[
            ConnectorCapability(
                capability_id="probe_filings_intel",
                title="Probe Filings Intelligence",
                description="Check filings-intelligence readiness and optionally run a live probe against the configured filing backbone.",
            ),
        ],
        config_schema={"optional": ["sample_symbol", "execute_live"]},
    ),
    "estimates_or_earnings_intel": IntegrationDefinition(
        integration_type="estimates_or_earnings_intel",
        title="Estimates / Earnings Intelligence",
        description="Credential-aware earnings and estimates-intelligence probe.",
        capabilities=[
            ConnectorCapability(
                capability_id="probe_estimates_intel",
                title="Probe Estimates Intelligence",
                description="Check earnings-estimates readiness and optionally run a live probe against the configured enrichment source.",
            ),
        ],
        config_schema={"optional": ["sample_symbol", "execute_live"]},
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
