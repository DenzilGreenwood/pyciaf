"""
Time and locale determinism utilities for CIAF core operations.

Ensures canonical timestamp generation, locale-independent operations,
and deterministic time-based computations for audit reproducibility.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import locale
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from .crypto import sha256_hash


class DeterministicClock:
    """
    Deterministic clock for reproducible timestamp generation.
    
    Provides canonical UTC timestamps and deterministic time-based
    operations for audit reproducibility.
    """
    
    def __init__(self, fixed_time: Optional[datetime] = None):
        """
        Initialize deterministic clock.
        
        Args:
            fixed_time: Optional fixed time for testing/reproducibility.
                       If None, uses current UTC time.
        """
        self._fixed_time = fixed_time
        self._call_count = 0
    
    def now(self) -> datetime:
        """Get current deterministic timestamp."""
        if self._fixed_time:
            # For deterministic testing, add microseconds based on call count
            # to ensure unique timestamps within a test run
            self._call_count += 1
            return self._fixed_time.replace(microsecond=self._call_count % 1000000)
        else:
            return datetime.now(timezone.utc)
    
    def now_iso(self) -> str:
        """Get current timestamp as ISO 8601 string."""
        return self.canonical_iso_format(self.now())
    
    def unix_timestamp(self) -> float:
        """Get current timestamp as Unix timestamp."""
        return self.now().timestamp()
    
    def unix_timestamp_int(self) -> int:
        """Get current timestamp as integer Unix timestamp."""
        return int(self.unix_timestamp())
    
    @staticmethod
    def canonical_iso_format(dt: datetime) -> str:
        """
        Convert datetime to canonical ISO 8601 format.
        
        Always uses UTC timezone and microsecond precision for determinism.
        
        Args:
            dt: Datetime to format
            
        Returns:
            Canonical ISO 8601 string
        """
        # Ensure UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            dt = dt.astimezone(timezone.utc)
        
        # Format with microsecond precision and 'Z' suffix
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    @staticmethod
    def parse_canonical_iso(iso_string: str) -> datetime:
        """
        Parse canonical ISO 8601 string to datetime.
        
        Args:
            iso_string: ISO 8601 formatted string
            
        Returns:
            Datetime object in UTC
        """
        # Handle both 'Z' and '+00:00' suffixes
        if iso_string.endswith('Z'):
            iso_string = iso_string[:-1] + '+00:00'
        
        return datetime.fromisoformat(iso_string).astimezone(timezone.utc)
    
    def time_hash(self, additional_data: str = "") -> str:
        """
        Generate deterministic hash based on current time and optional data.
        
        Args:
            additional_data: Additional data to include in hash
            
        Returns:
            SHA256 hash string
        """
        timestamp_str = self.now_iso()
        combined_data = f"{timestamp_str}:{additional_data}".encode('utf-8')
        return sha256_hash(combined_data)


class LocaleIndependentOps:
    """
    Locale-independent operations for deterministic text processing.
    
    Ensures consistent behavior across different system locales.
    """
    
    @staticmethod
    def normalize_string(s: str) -> str:
        """
        Normalize string for locale-independent comparison.
        
        Args:
            s: String to normalize
            
        Returns:
            Normalized string
        """
        # Convert to lowercase and strip whitespace
        normalized = s.lower().strip()
        
        # Remove any locale-specific characters that might cause issues
        # This is a simple implementation - could be extended based on needs
        return normalized
    
    @staticmethod
    def compare_strings(s1: str, s2: str) -> int:
        """
        Locale-independent string comparison.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            -1 if s1 < s2, 0 if s1 == s2, 1 if s1 > s2
        """
        norm1 = LocaleIndependentOps.normalize_string(s1)
        norm2 = LocaleIndependentOps.normalize_string(s2)
        
        if norm1 < norm2:
            return -1
        elif norm1 > norm2:
            return 1
        else:
            return 0
    
    @staticmethod
    def sort_strings(strings: list[str]) -> list[str]:
        """
        Locale-independent string sorting.
        
        Args:
            strings: List of strings to sort
            
        Returns:
            Sorted list of strings
        """
        return sorted(strings, key=LocaleIndependentOps.normalize_string)
    
    @staticmethod
    def format_number(number: float, decimal_places: int = 6) -> str:
        """
        Locale-independent number formatting.
        
        Always uses '.' as decimal separator regardless of system locale.
        
        Args:
            number: Number to format
            decimal_places: Number of decimal places
            
        Returns:
            Formatted number string
        """
        return f"{number:.{decimal_places}f}"
    
    @staticmethod
    def with_c_locale(func):
        """
        Decorator to execute function with C locale for determinism.
        
        Args:
            func: Function to execute
            
        Returns:
            Decorator function
        """
        def wrapper(*args, **kwargs):
            # Save current locale
            current_locale = locale.getlocale()
            
            try:
                # Set C locale for deterministic behavior
                locale.setlocale(locale.LC_ALL, 'C')
                return func(*args, **kwargs)
            finally:
                # Restore original locale
                try:
                    locale.setlocale(locale.LC_ALL, current_locale)
                except locale.Error:
                    # Fallback if original locale can't be restored
                    pass
        
        return wrapper


class DeterministicTimestampGenerator:
    """
    Generates deterministic timestamps for audit operations.
    
    Combines deterministic clock with additional entropy sources
    for unique, reproducible timestamps.
    """
    
    def __init__(self, clock: Optional[DeterministicClock] = None, base_entropy: str = "ciaf"):
        """
        Initialize timestamp generator.
        
        Args:
            clock: Deterministic clock instance. If None, creates default.
            base_entropy: Base entropy string for deterministic variation
        """
        self.clock = clock or DeterministicClock()
        self.base_entropy = base_entropy
        self.sequence_counter = 0
    
    def generate_timestamp(self, operation_id: str = "", additional_entropy: str = "") -> str:
        """
        Generate deterministic timestamp for an operation.
        
        Args:
            operation_id: Identifier for the operation
            additional_entropy: Additional entropy for uniqueness
            
        Returns:
            Canonical ISO 8601 timestamp string
        """
        self.sequence_counter += 1
        
        # Combine all entropy sources
        entropy_parts = [
            self.base_entropy,
            operation_id,
            additional_entropy,
            str(self.sequence_counter)
        ]
        
        combined_entropy = ":".join(part for part in entropy_parts if part)
        
        # Get base timestamp
        base_time = self.clock.now()
        
        # Add deterministic microsecond offset based on entropy
        entropy_hash = sha256_hash(combined_entropy.encode('utf-8'))
        microsecond_offset = int(entropy_hash[:6], 16) % 1000000
        
        # Create final timestamp
        final_time = base_time.replace(microsecond=microsecond_offset)
        
        return self.clock.canonical_iso_format(final_time)
    
    def generate_timestamped_id(self, prefix: str = "id", operation_id: str = "") -> str:
        """
        Generate timestamped identifier.
        
        Args:
            prefix: Prefix for the identifier
            operation_id: Operation context for determinism
            
        Returns:
            Timestamped identifier string
        """
        timestamp = self.generate_timestamp(operation_id)
        # Convert timestamp to compact format for ID
        compact_timestamp = timestamp.replace('-', '').replace(':', '').replace('.', '').replace('Z', '')
        return f"{prefix}_{compact_timestamp}"


# Global instances for convenience
default_clock = DeterministicClock()
default_timestamp_generator = DeterministicTimestampGenerator()


# Convenience functions
def now_iso() -> str:
    """Get current timestamp as canonical ISO 8601 string."""
    return default_clock.now_iso()


def canonical_timestamp(dt: Optional[datetime] = None) -> str:
    """Get canonical timestamp for given datetime or current time."""
    if dt is None:
        dt = default_clock.now()
    return DeterministicClock.canonical_iso_format(dt)


def deterministic_timestamp(operation_id: str = "", additional_entropy: str = "") -> str:
    """Generate deterministic timestamp for an operation."""
    return default_timestamp_generator.generate_timestamp(operation_id, additional_entropy)


def timestamped_id(prefix: str = "id", operation_id: str = "") -> str:
    """Generate timestamped identifier."""
    return default_timestamp_generator.generate_timestamped_id(prefix, operation_id)


def normalize_for_determinism(text: str) -> str:
    """Normalize text for deterministic processing."""
    return LocaleIndependentOps.normalize_string(text)


def compare_deterministic(s1: str, s2: str) -> int:
    """Locale-independent string comparison."""
    return LocaleIndependentOps.compare_strings(s1, s2)


def sort_deterministic(strings: list[str]) -> list[str]:
    """Locale-independent string sorting."""
    return LocaleIndependentOps.sort_strings(strings)


# Context managers for deterministic operations

class FixedTimeContext:
    """Context manager for operations with fixed time."""
    
    def __init__(self, fixed_time: datetime):
        self.fixed_time = fixed_time
        self.original_clock = None
    
    def __enter__(self):
        global default_clock, default_timestamp_generator
        self.original_clock = default_clock
        default_clock = DeterministicClock(self.fixed_time)
        default_timestamp_generator = DeterministicTimestampGenerator(default_clock)
        return default_clock
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global default_clock, default_timestamp_generator
        default_clock = self.original_clock
        default_timestamp_generator = DeterministicTimestampGenerator(default_clock)


class DeterministicContext:
    """Context manager for fully deterministic operations."""
    
    def __init__(self, fixed_time: datetime, base_entropy: str = "test"):
        self.fixed_time = fixed_time
        self.base_entropy = base_entropy
        self.original_clock = None
        self.original_generator = None
    
    def __enter__(self):
        global default_clock, default_timestamp_generator
        self.original_clock = default_clock
        self.original_generator = default_timestamp_generator
        
        default_clock = DeterministicClock(self.fixed_time)
        default_timestamp_generator = DeterministicTimestampGenerator(default_clock, self.base_entropy)
        
        return (default_clock, default_timestamp_generator)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global default_clock, default_timestamp_generator
        default_clock = self.original_clock
        default_timestamp_generator = self.original_generator