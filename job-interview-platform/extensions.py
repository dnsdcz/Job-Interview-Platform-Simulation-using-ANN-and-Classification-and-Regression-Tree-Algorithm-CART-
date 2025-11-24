# extensions.py
import logging
import os
import warnings

from flask_mail import Mail
from flask_mysqldb import MySQL
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("job-interview-platform")

# suppress noisy warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# flask extensions
mysql = MySQL()
mail = Mail()
limiter = Limiter(key_func=get_remote_address,
                  default_limits=["10 per minute"])

# NLP models (global singletons)
try:
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("✅ SentenceTransformer loaded.")
except Exception as e:
    sentence_model = None
    logger.error(f"❌ Failed to load SentenceTransformer: {e}")

try:
    kw_model = KeyBERT("all-MiniLM-L6-v2")
    logger.info("✅ KeyBERT loaded.")
except Exception as e:
    kw_model = None
    logger.error(f"⚠️ Failed to load KeyBERT: {e}")
