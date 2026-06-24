from flask import Flask
from config import Config
from extensions import mysql, mail, limiter, logger
from blueprints import register_blueprints
from blueprints.admin import admin_bp


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)
    app.register_blueprint(admin_bp)

    mysql.init_app(app)
    mail.init_app(app)
    limiter.init_app(app)

    register_blueprints(app)

    @app.route("/debug-routes")
    def debug_routes():
        lines = []
        for rule in app.url_map.iter_rules():
            lines.append(f"{rule.endpoint} -> {rule}")
        return "<br>".join(lines)

    logger.info("✅ Application created and blueprints registered.")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
