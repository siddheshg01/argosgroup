"""Discover source columns and map recognized concepts without requiring names."""
import re
import pandas as pd

ALIASES = {
    "order_id": ("order_id", "orderid", "transaction_id", "transactionid", "invoice_id", "receipt_id"),
    "date": ("order_date", "orderdate", "transaction_date", "sale_date", "date", "created_at"),
    "customer_id": ("customer_id", "customerid", "client_id", "buyer_id"),
    "customer_name": ("customer_name", "customername", "client_name", "buyer_name"),
    "location": ("customer_location", "location", "city", "state", "country", "region"),
    "region": ("region", "state", "province"),
    "product": ("product_name", "productname", "product", "item_name", "sku_name"),
    "product_id": ("product_id", "productid", "sku", "item_id"),
    "category": ("category", "product_category"),
    "subcategory": ("sub_category", "subcategory", "product_subcategory"),
    "price": ("unit_price", "unitprice", "price", "selling_price", "unit_selling_price"),
    "quantity": ("quantity", "units_sold", "qty"),
    "revenue": ("revenue", "net_sales", "netsales", "total_sales", "totalsales", "sales"),
    "order_amount": ("total_amount", "totalamount", "order_amount", "order_total", "grand_total", "amount"),
    "profit": ("profit", "net_profit", "gross_profit"),
    "cost": ("cost", "total_cost", "cost_of_goods_sold", "cogs"),
    "discount": ("discount", "discount_amount", "discount_rate"),
    "tax": ("tax", "tax_amount"),
    "shipping_cost": ("shipping_cost", "shippingcost", "shipping_fee"),
    "refund": ("refund", "refund_amount", "is_refunded"),
    "payment_method": ("payment_method", "paymentmethod", "payment_type"),
    "order_status": ("order_status", "orderstatus", "status"),
}

def _normalize(value: str) -> str:
    # Split CamelCase first so OrderID and order_id normalize identically.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value).strip())
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

def discover_schema(frame: pd.DataFrame) -> dict:
    """Return source-to-internal mappings and unavailable concepts.

    Exact normalized aliases are preferred. A source column is mapped at most
    once, so a dataset with both city and state retains the more specific city
    as ``location`` and leaves the other source column available for later use.
    """
    normalized = {_normalize(col): col for col in frame.columns}
    mapping = {}
    used = set()
    for internal, aliases in ALIASES.items():
        for alias in aliases:
            source = normalized.get(_normalize(alias))
            if source is not None and source not in used:
                mapping[internal] = source
                used.add(source)
                break
    return {
        "mapping": mapping,
        "unavailable": [key for key in ALIASES if key not in mapping],
        "source_columns": list(frame.columns),
        "unmapped_columns": [c for c in frame.columns if c not in used],
    }
