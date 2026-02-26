"""Kafka event publisher adapter for the Energy Sustainability service.

Wraps aumos-common's EventPublisher to provide domain-specific publish methods
and satisfies the IEventPublisher protocol used by core services.
"""

from typing import Any

from aumos_common.events import EventPublisher, KafkaSettings
from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Canonical topic names for this service
TOPIC_CARBON_TRACKED = "aumos.energy.carbon.tracked"
TOPIC_ROUTE_DECIDED = "aumos.energy.route.decided"
TOPIC_REPORT_GENERATED = "aumos.energy.report.generated"
TOPIC_OPTIMIZATIONS_GENERATED = "aumos.energy.optimizations.generated"


class EnergyEventPublisher:
    """Kafka event publisher for Energy Sustainability domain events.

    Wraps the aumos-common EventPublisher and provides typed publish methods
    for each domain event type.
    """

    def __init__(self, kafka_settings: KafkaSettings) -> None:
        """Initialise with Kafka broker configuration.

        Args:
            kafka_settings: Kafka broker and topic configuration from AumOS settings.
        """
        self._publisher = EventPublisher(kafka_settings)

    async def start(self) -> None:
        """Start the underlying Kafka producer.

        Must be called before any publish operations.
        """
        await self._publisher.start()
        logger.info("EnergyEventPublisher started")

    async def stop(self) -> None:
        """Flush pending messages and stop the Kafka producer."""
        await self._publisher.stop()
        logger.info("EnergyEventPublisher stopped")

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish a domain event to a Kafka topic.

        Satisfies the IEventPublisher protocol for dependency injection
        into core services.

        Args:
            topic: Kafka topic name.
            event: Event payload dict — will be serialized as JSON.
        """
        await self._publisher.publish(topic=topic, payload=event)
        logger.debug("Event published", topic=topic, event_keys=list(event.keys()))
