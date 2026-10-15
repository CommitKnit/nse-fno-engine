"""config — runtime settings (``settings``, loaded from .env) and static constants."""

from config.settings import settings
from config.constants import NIFTY, BANKNIFTY

__all__ = ["settings", "NIFTY", "BANKNIFTY"]