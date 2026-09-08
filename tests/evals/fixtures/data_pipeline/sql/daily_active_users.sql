-- Rebuilt in full every run. No incremental logic; the raw table is small today.
create or replace table marts.daily_active_users as
select
    date_trunc('day', occurred_at) as day,
    count(distinct user_id) as active_users
from raw.events
group by 1;
