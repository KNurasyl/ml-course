-- PostgreSQL schema of the fictional company «Лампа и Чайник» (data/company, R07). Load with:
--   psql -d company -f data/company/schema.sql
--   \copy products FROM 'data/company/products.csv' CSV HEADER   (the same for customers, orders, order_items)
CREATE TABLE products (sku text PRIMARY KEY, name text NOT NULL, category text NOT NULL, category_title text NOT NULL,
                       price_kzt integer NOT NULL CHECK (price_kzt > 0), warranty_months integer NOT NULL);
CREATE TABLE customers (customer_id text PRIMARY KEY, city text NOT NULL, segment text NOT NULL CHECK (segment IN ('b2c', 'b2b')),
                        registered_on date NOT NULL);
CREATE TABLE orders (order_id text PRIMARY KEY, customer_id text NOT NULL REFERENCES customers, city text NOT NULL,
                     created_at date NOT NULL, status text NOT NULL, payment_method text NOT NULL, delivery_type text NOT NULL,
                     delivery_price_kzt integer NOT NULL, items_total_kzt integer NOT NULL, delivered_at date,
                     promised_days_max integer NOT NULL);
CREATE TABLE order_items (order_id text NOT NULL REFERENCES orders, sku text NOT NULL REFERENCES products,
                          qty integer NOT NULL CHECK (qty > 0), price_kzt integer NOT NULL);
