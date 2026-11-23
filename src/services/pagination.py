import math
from typing import List, TypeVar

from src.schemas import PaginatedMetaData, PaginatedResult

T = TypeVar("T")


def paginate(items: List[T], page: int, page_size: int) -> PaginatedResult[T]:
    """Slices `items` into the requested page. `page` is 1-based.

    An out-of-range page returns empty results (it is not clamped to the last page).
    Raises ValueError if `page` < 1.
    """
    if page < 1:
        raise ValueError("page must be >= 1")

    total_items = len(items)
    total_pages = math.ceil(total_items / page_size) if page_size > 0 else 0

    start = (page - 1) * page_size
    end = start + page_size

    metadata = PaginatedMetaData(
        total_items=total_items,
        total_pages=total_pages,
        current_page=page,
        page_size=page_size,
        has_next=page < total_pages,
        has_prev=page > 1,
    )
    return PaginatedResult(results=items[start:end], pagination=metadata)
