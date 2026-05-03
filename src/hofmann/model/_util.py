"""Shared serialisation helpers for model dataclasses."""

from __future__ import annotations

import dataclasses

from hofmann.model.composition import Composition

SiteContent = str | Composition

_field_defaults_cache: dict[tuple[type, frozenset[str]], dict] = {}


def _field_defaults(cls: type, *, exclude: frozenset[str] = frozenset()) -> dict:
    """Return a dict of ``{field_name: default}`` for a dataclass.

    Only fields with simple defaults (not ``MISSING`` and not
    ``default_factory``) are included.  Fields listed in *exclude*
    are skipped.  This is used by ``to_dict()`` methods to compare
    current values against defaults so only non-default fields are
    serialised.  Results are cached per ``(cls, exclude)`` pair.
    """
    key = (cls, exclude)
    if key not in _field_defaults_cache:
        # Fields with no default and fields using default_factory both
        # have f.default == MISSING, so this single check excludes both.
        _field_defaults_cache[key] = {
            f.name: f.default
            for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING
            and f.name not in exclude
        }
    return _field_defaults_cache[key]


def _site_species(site: SiteContent) -> frozenset[str]:
    """Return the set of constituent species at a site row.

    For a plain string, returns a singleton frozenset.  For a
    :class:`Composition`, returns its constituent species (vacancy
    excluded).

    Args:
        site: A site content value: either a species label string or
            a :class:`Composition`.

    Returns:
        A frozenset of constituent species labels.
    """
    if isinstance(site, Composition):
        return site.species
    return frozenset({site})
