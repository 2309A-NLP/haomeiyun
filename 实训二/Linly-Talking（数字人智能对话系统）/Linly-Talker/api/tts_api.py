from src.api.tts_api import app


if __name__ == "__main__":
    import runpy

    runpy.run_module("src.api.tts_api", run_name="__main__")
