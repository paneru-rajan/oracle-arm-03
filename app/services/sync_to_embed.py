import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

import asyncpg

from config import settings
from services.embedder import embedder
from services.vector_store import chat_store

logger = logging.getLogger(__name__)

SYNC_QUERY = """
    SELECT 
        m._id AS message_id,
        m.content AS question,
        COALESCE(ar."editedResponse", ar."suggestedResponse") AS answer,
        bg.id AS guest_id,
        b.home_id AS property_id,
        h.workspace_id AS host_id,
        ar."messageCategory" AS category,
        ar.created_at AS created_at,
        ar.updated_at AS updated_at
    FROM 
        public.ai_response ar
    JOIN 
        public.message m ON ar."messageId" = m._id
    JOIN 
        public.booking b ON ar."bookingId" = b.id
    JOIN 
        public.home h ON b.home_id = h._id
    LEFT JOIN 
         public.booking_guest bg ON b.id = bg."bookingId" AND bg."isMainGuest" = true
    WHERE 
        (ar."responseStatus" IN ('approved', 'edited') OR ar."responseSentAt" IS NOT NULL)
        AND m."senderType" = 'guest'
        AND ar.updated_at > $1
    ORDER BY ar.updated_at ASC
    LIMIT 1000;
"""

BATCH_SIZE = 1 # 

def _ensure_timezone(dt: datetime) -> datetime:
    """Ensure datetime object has timezone information (UTC if missing)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

async def sync_chat_data():
    """
    Periodic job to sync chat data from Postgres to Elasticsearch.
    """
    if not settings.database_url:
        logger.warning("DATABASE_URL not set. Skipping chat sync.")
        return

    logger.info("Starting chat data sync...")

    try:
        last_ts_ms = await chat_store.get_latest_timestamp()
        # Convert to UTC aware then strip tzinfo to match Postgres timestamp without time zone
        last_sync_dt = datetime.fromtimestamp(last_ts_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)
        
        logger.info(f"Last synced timestamp: {last_ts_ms} ({last_sync_dt} UTC naive)")

        conn = await asyncpg.connect(settings.database_url)

        try:
            rows = await conn.fetch(SYNC_QUERY, last_sync_dt)
            total_rows = len(rows)
            logger.info(f"Fetched {total_rows} new records to sync.")

            if not rows:
                return

            for i in range(0, total_rows, BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                
                texts_to_embed: List[str] = []
                docs_to_index: List[Dict[str, Any]] = []

                for row in batch:
                    question = row['question'] or ""
                    answer = row['answer'] or ""
                    
                    texts_to_embed.append(f"Q: {question}\nA: {answer}")
                    
                    updated_at = _ensure_timezone(row['updated_at'])
                    created_at = _ensure_timezone(row['created_at'])
                    
                    docs_to_index.append({
                        "_id": str(row['message_id']),
                        "message_id": str(row['message_id']),
                        "question": question,
                        "answer": answer,
                        "guest_id": str(row['guest_id']) if row['guest_id'] else None,
                        "property_id": str(row['property_id']),
                        "host_id": str(row['host_id']),
                        "category": str(row['category']) if row['category'] else None,
                        "created_at": created_at.isoformat(),
                        "updated_at": updated_at.isoformat()
                    })
                
                embeddings = await embedder.embed(texts_to_embed, settings.default_model_type)
                
                for doc, emb in zip(docs_to_index, embeddings):
                    doc["embedding"] = emb
                
                await chat_store.bulk_index(docs_to_index)
                
            logger.info(f"Successfully synced {total_rows} records.")

        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Error during chat sync: {e}", exc_info=True)

async def start_scheduler():
    """
    Background task that runs the sync job periodically.
    """
    logger.info("Scheduler started. Waiting 5 minutes before first sync...")
    await asyncio.sleep(1)  # 5 minutes initial delay
    
    while True:
        try:
            await sync_chat_data()
        except Exception as e:
            logger.error(f"Scheduler crashed: {e}")
        
        wait_seconds = settings.sync_interval_minutes * 60
        logger.info(f"Next sync in {settings.sync_interval_minutes} minutes.")
        await asyncio.sleep(wait_seconds)
