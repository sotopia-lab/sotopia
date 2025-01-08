To deploy the Streamlit UI to Modal, run the following command:
```bash
cd ui/streamlit_ui
modal deploy modal_streamlit_app.py
```

To serve the Streamlit UI, run the following command:
```bash
modal serve modal_streamlit_app.py
```

Before deploying the Streamlit UI, do check the `API_BASE` and `WS_BASE` in the `streamlit_ui/app.py` and set to your API server's endpoint (which could either be local or your Modal endpoint).

## Streamlit UI
To run the Streamlit UI, run the following command:
```bash
cd sotopia/ui/streamlit_ui
uv run streamlit run app.py
```
