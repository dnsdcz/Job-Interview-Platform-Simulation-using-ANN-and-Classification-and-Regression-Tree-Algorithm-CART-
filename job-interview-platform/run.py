# run.py
from flask import Flask
from config import Config
from extensions import mysql, mail, limiter, logger
from blueprints import register_blueprints  # ⬅️ this will handle ALL blueprints


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    # init extensions
    mysql.init_app(app)
    mail.init_app(app)
    limiter.init_app(app)

    # register ALL blueprints in one place
    register_blueprints(app)

    logger.info("✅ Application created and blueprints registered.")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
