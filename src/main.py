from myapp import app
from database_from_snapshot import create_from_snapshot

if __name__ == "__main__":
    create_from_snapshot()
    app.run()
