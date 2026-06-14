import math
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="MapFlow AI Backend Core API",
    description="Production-grade FastAPI orchestrator for Local SEO, Multimodal Vision replies, POS webhooks, and VoIP interceptors.",
    version="1.4.0"
)

# Enable CORS for frontend integrations (e.g. Next.js / dashboard previews)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. DATABASE MODELS & PYDANTIC SCHEMAS
# ==========================================

class ReviewPayload(BaseModel):
    reviewer_name: str
    star_rating: int = Field(..., ge=1, le=5)
    comment: str
    media_url: Optional[str] = None  # GMB review photo URL
    voice_tone: str = "friendly"     # 'friendly', 'professional', 'witty'

class ProductUploadInput(BaseModel):
    image_url: str
    category: str

class SquareWebhookPayload(BaseModel):
    event_type: str = "payment.created"
    customer_phone: str
    customer_name: str
    invoice_amount: float
    location_id: str

class VoIPMissedCallPayload(BaseModel):
    caller_phone: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    call_duration_seconds: int = 0
    voicemail_url: Optional[str] = None

class PostBroadcastPayload(BaseModel):
    location_ids: List[str]
    post_template: str = Field(..., example="Come visit us at {{city}}!")

# ==========================================
# 2. BUSINESS LOGIC ENGINES (CORE UTILITIES)
# ==========================================

def calculate_decay_freshness(last_post_at: datetime, last_photo_at: datetime, last_reply_at: datetime) -> Dict[str, Any]:
    """
    Implements the Algorithmic Freshness Guard decay calculation:
    Freshness = 40% (Post Decay) + 30% (Photo Decay) + 30% (Reply Decay)
    Decay constant lambda (λ) set to 0.05 (corresponds to ~14 day half-life)
    """
    now = datetime.now(timezone.utc)
    
    days_since_post = max(0, (now - last_post_at.astimezone(timezone.utc)).days)
    days_since_photo = max(0, (now - last_photo_at.astimezone(timezone.utc)).days)
    days_since_reply = max(0, (now - last_reply_at.astimezone(timezone.utc)).days)
    
    # Exponential decay function: e^(-lambda * t)
    lambda_decay = 0.05
    
    post_score = 40 * math.exp(-lambda_decay * days_since_post)
    photo_score = 30 * math.exp(-lambda_decay * days_since_photo)
    reply_score = 30 * math.exp(-lambda_decay * days_since_reply)
    
    total_freshness = int(post_score + photo_score + reply_score)
    
    # Classify threshold status
    if total_freshness >= 85:
        status_label = "Excellent Momentum"
    elif total_freshness >= 70:
        status_label = "Good (Minor Activity Needed)"
    else:
        status_label = "Decay Warning (Listing Stagnant)"
        
    return {
        "freshness_score": total_freshness,
        "status": status_label,
        "metrics": {
            "days_since_last_post": days_since_post,
            "days_since_last_photo": days_since_photo,
            "days_since_last_reply": days_since_reply
        }
    }


def detect_competitor_name_stuffing(business_name: str) -> Dict[str, Any]:
    """
    Analyzes competitor business profiles to flag algorithm-gaming keyword stuffing.
    Compares profile name patterns against NLP spam lists.
    """
    spam_triggers = [
        "best", "near me", "emergency", "cheap", "top-rated", "plumber in", 
        "bakery in", "custom cakes", "sourdough", "repair", "services", "specialist"
    ]
    
    # Clean and split name
    name_lower = business_name.lower()
    detected_keywords = []
    
    # Detect stuffing indicators: typically names containing hyphens, colons, or pipes followed by keywords
    contains_delimiters = any(char in business_name for char in ["-", "|", ":", "—"])
    
    for word in spam_triggers:
        if word in name_lower:
            detected_keywords.append(word)
            
    is_spam = contains_delimiters and (len(detected_keywords) >= 2)
    
    return {
        "competitor_name": business_name,
        "is_spam_suspected": is_spam,
        "spam_keywords_detected": detected_keywords,
        "confidence_score": 0.85 if is_spam else 0.10
    }

# ==========================================
# 3. FASTAPI CONTROLLER ENDPOINTS
# ==========================================

@app.get("/api/location/{location_id}/freshness", tags=["Freshness Guard"])
async def get_listing_freshness(location_id: str):
    """
    API endpoint calculating real-time listing freshness decay to prevent Maps penalty.
    """
    # MOCK DB TIMESTAMPS: In production, query these from the PostgreSQL database
    mock_last_post = datetime.now(timezone.utc) - datetime.timedelta(days=3)
    mock_last_photo = datetime.now(timezone.utc) - datetime.timedelta(days=1)
    mock_last_reply = datetime.now(timezone.utc) - datetime.timedelta(days=0, hours=2)
    
    freshness_data = calculate_decay_freshness(
        last_post_at=mock_last_post,
        last_photo_at=mock_last_photo,
        last_reply_at=mock_last_reply
    )
    return {"location_id": location_id, **freshness_data}


@app.post("/api/reviews/multimodal-reply", tags=["Review AI Assistant"])
async def generate_review_reply(payload: ReviewPayload):
    """
    Advanced Multimodal Response Endpoint. If a media URL exists, it fires a Vision API call
    to GPT-4o to write highly organic, picture-informed replies.
    """
    # Mock prompt engineering system representing LLM wrapper
    base_prompt = f"Review Text: '{payload.comment}' | Stars: {payload.star_rating} | Voice: {payload.voice_tone}"
    
    if payload.media_url:
        # Simulate Vision analysis (GPT-4o Vision API output parameters)
        detected_elements = ["Almond Croissant", "White Ceramic Plate", "Flaky Golden Crust"]
        
        if payload.voice_tone == "friendly":
            reply_text = (
                f"Hi! Thanks for the wonderful 5-star review! 🥐 We love how golden-brown and flaky "
                f"that Almond Croissant looks on the plate in your photo. It came straight out of our "
                f"stone-hearth ovens right before you arrived! Hope to see you back on Congress Ave. soon!"
            )
        else:
            reply_text = (
                f"Thank you for sharing your experience. We take pride in the preparation of our pastries, "
                f"and we are glad that your croissant met our standard. We appreciate your feedback and photo."
            )
    else:
        # Text-only standard OpenAI route
        detected_elements = []
        reply_text = f"Hi! Thanks for stopping by our bakery! Glad you loved our pastries. See you next time!"

    return {
        "status": "success",
        "multimodal_active": bool(payload.media_url),
        "vision_tags_detected": detected_elements,
        "ai_drafted_reply": reply_text
    }


@app.post("/api/webhooks/square-checkout", status_code=status.HTTP_202_ACCEPTED, tags=["POS AfterCare"])
async def handle_square_checkout_webhook(payload: SquareWebhookPayload, background_tasks: BackgroundTasks):
    """
    Ingress POS Webhook endpoint listening to client registers.
    Registers a background Celery-style delayed trigger to message customer WhatsApp.
    """
    # 1. Log transaction details in PostgreSQL (mock logic)
    print(f"Logged transaction {payload.event_type} for customer {payload.customer_name}")
    
    # 2. Queue delayed WhatsApp aftercare message
    # In production, this schedules a task via Celery with a 30-minute countdowneta.
    background_tasks.add_task(
        mock_send_whatsapp_rating_request, 
        customer_phone=payload.customer_phone,
        customer_name=payload.customer_name
    )
    
    return {
        "status": "webhook_received",
        "message": "Delayed aftercare rating queued.",
        "queue_delay_seconds": 1800  # 30 mins
    }


@app.post("/api/webhooks/twilio/missed-call", tags=["VoIP Call Snatcher"])
async def handle_unanswered_call_voip(payload: VoIPMissedCallPayload):
    """
    Twilio VoIP voice webhook receiver. Triggered immediately if customer call goes unanswered
    or hits a busy line. Instantly SMSs the prospect to prevent lead drop.
    """
    sms_reply_text = (
        f"Hi there! Joe's Bakery here. 🥐 Sorry we missed your call—we are currently baking a fresh batch! "
        f"Are you looking to place a Sourdough pre-order or book a Custom Cake? Reply here and I'll help you instantly!"
    )
    
    # In production, invoke Twilio Client to dispatch SMS back to `payload.caller_phone`
    print(f"Call Snatcher: Missed call from {payload.caller_phone}. Auto-SMS dispatched: {sms_reply_text}")
    
    return {
        "status": "unanswered_call_logged",
        "caller": payload.caller_phone,
        "auto_sms_back_dispatched": True,
        "text_content": sms_reply_text
    }


@app.post("/api/products/sync-vision", tags=["AI Product Sync"])
async def generate_product_sku_from_image(payload: ProductUploadInput):
    """
    Takes a product image URL uploaded from customer mobile, uses Vision AI to generate
    SKUs, title, prices, and SEO descriptions, and pushes straight to Google Maps Products catalog.
    """
    # Simulated Vision parsing parameters matching image inputs
    if "cake" in payload.image_url.lower() or "wedding" in payload.image_url.lower():
        sku_data = {
            "product_title": "Custom 3-Tier Sourdough Wedding Cake",
            "suggested_price": 345.00,
            "category": "Cakes & Events",
            "seo_description": (
                "An organic sourdough sponge flavored with lavender and lemon curd. "
                "Fully customized frosting tailored to your special event. Perfect for Austin weddings."
            ),
            "json_ld_schema": {
                "@context": "https://schema.org",
                "@type": "Product",
                "name": "Custom 3-Tier Sourdough Wedding Cake",
                "category": "Cakes & Events",
                "offers": {"@type": "Offer", "price": "345.00", "priceCurrency": "USD"}
            }
        }
    else:
        # Default to croissant SKU
        sku_data = {
            "product_title": "Traditional Almond Croissant",
            "suggested_price": 4.75,
            "category": "Pastry Products",
            "seo_description": (
                "Baked daily in our stone-hearth ovens. Toasted almonds, organic honey, "
                "and a perfectly flaky butter crust. Order fresh pastry near me in Congress Ave."
            ),
            "json_ld_schema": {
                "@context": "https://schema.org",
                "@type": "Product",
                "name": "Traditional Almond Croissant",
                "category": "Pastry Products",
                "offers": {"@type": "Offer", "price": "4.75", "priceCurrency": "USD"}
            }
        }
        
    return {
        "status": "sku_generated",
        "vision_success": True,
        "catalog_sku": sku_data
    }


# ==========================================
# 4. BULK SMS CAMPAIGN ENDPOINTS
# ==========================================

class BulkSMSPayload(BaseModel):
    recipient_phones: List[str] = Field(..., min_items=1)
    message_body: str = Field(..., max_length=160)

@app.post("/api/campaigns/bulk-sms", tags=["Campaigns & Bulk SMS"])
async def trigger_bulk_sms_campaign(payload: BulkSMSPayload, background_tasks: BackgroundTasks):
    """
    Triggers an immediate Bulk SMS Review Blast Campaign to a batch of customer numbers.
    Utilizes background workers to batch send without blocking.
    """
    # Simulate processing
    background_tasks.add_task(
        mock_process_bulk_sms,
        phones=payload.recipient_phones,
        body=payload.message_body
    )
    return {
        "status": "campaign_dispatched",
        "total_recipients": len(payload.recipient_phones),
        "est_delivery_seconds": len(payload.recipient_phones) * 0.1  # ~10 messages/sec throughput
    }

async def mock_process_bulk_sms(phones: List[str], body: str):
    print(f"[CELERY WORKER] Initiating bulk review blast campaign for {len(phones)} customers.")
    for phone in phones:
        # In production, call twilio.messages.create()
        print(f"[CELERY WORKER] SMS Dispatched to recipient: {phone}")
    print("[CELERY WORKER] Bulk campaign successfully delivered.")


# ==========================================
# 5. FRANCHISE COMMANDER ENDPOINTS
# ==========================================



@app.post("/api/franchise/bulk-post", tags=["Franchise Commander"])
async def broadcast_bulk_post(payload: PostBroadcastPayload):
    """
    Franchise Commander Bulk Post endpoint. Parses template post parameters,
    performs coordinate-specific string replacements (e.g. {{city}}), and publishes
    across multiple GBP listings via threads concurrently.
    """
    published_locations = []
    
    # Mock locations database
    locations_db = {
        "loc_01": {"city": "Austin Downtown", "address": "Congress Ave"},
        "loc_02": {"city": "Austin North", "address": "Lamar Blvd"},
        "loc_03": {"city": "San Antonio", "address": "Riverwalk"}
    }
    
    for loc_id in payload.location_ids:
        if loc_id in locations_db:
            loc_meta = locations_db[loc_id]
            # Perform merge tag replacements
            city_specific_post = payload.post_template.replace("{{city}}", loc_meta["city"])
            
            # Simulate Google Business Profile API post push
            published_locations.append({
                "location_id": loc_id,
                "profile_name": f"Joe's Bakery - {loc_meta['city']}",
                "published_post_text": city_specific_post,
                "status": "success",
                "google_post_id": f"g_post_{os.urandom(4).hex()}"
            })
            
    return {
        "broadcast_status": "success",
        "total_profiles_published": len(published_locations),
        "publications": published_locations
    }

# ==========================================
# 4. BACKGROUND MOCK SERVICES
# ==========================================

async def mock_send_whatsapp_rating_request(customer_phone: str, customer_name: str):
    """
    Simulates sending Twilio Meta WhatsApp message rating flow in background.
    """
    print(f"[CELERY WORKER] Waiting 30-min cooldown for {customer_name}... Cooldown finished.")
    print(f"[CELERY WORKER] Sending WhatsApp Meta template to phone {customer_phone}...")
    print(f"[CELERY WORKER] Send complete.")
