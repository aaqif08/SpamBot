"""BotShield AI — framework-independent machine-learning package.

Nothing in this package imports FastAPI or SQLAlchemy. The HTTP layer in
``app/`` calls into this package; this package never calls back.
"""

FEATURE_VERSION = "paper-31-v1"
