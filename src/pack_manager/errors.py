"""Typed failures for bad pack-event inputs. These are not UNCERTAIN verdicts."""


class PackManagerError(Exception):
    """Base error for Pack Manager."""


class InvalidInputError(PackManagerError):
    """Order, catalogue, or photo payload cannot be processed."""


class CorruptImageError(InvalidInputError):
    """An evidence photo cannot be decoded."""


class MissingCatalogueError(InvalidInputError):
    """An ordered SKU has no catalogue entry."""
