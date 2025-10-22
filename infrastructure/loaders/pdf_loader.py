from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


class PDFLoaderAdapter:
    """Adapter de loader de PDF usando LangChain (PyMuPDFLoader)."""

    def load(self, filepath: str) -> list[Document]:
        loader = PyMuPDFLoader(filepath)
        return loader.load()
