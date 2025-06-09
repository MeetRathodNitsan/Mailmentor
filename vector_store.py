from llama_index.core import Document, VectorStoreIndex
from transformers import AutoTokenizer, AutoModel

def initialize_vector_store(self):
    try:
        # Ensure the embedding model is initialized
        if not self.embed_model:
            raise ValueError("Embedding model is not initialized")
        
        # Fetch emails
        emails = self.fetch_emails(1000)  # Fetch more emails for better search
        if not emails:
            raise ValueError("No emails fetched for vector store initialization")
        
        documents = []
        for email in emails:
            # Validate email fields
            if not email.get('subject') or not email.get('sender') or not email.get('timestamp'):
                raise ValueError(f"Invalid email data: {email}")
            
            # Create document text
            text = f"""
            Subject: {email['subject']}
            From: {email['sender']}
            Date: {email['timestamp']}
            
            Content:
            {email.get('body', '')}
            
            Summary:
            {email.get('summary', '')}
            """
            
            # Create document
            doc = Document(
                text=text,
                metadata={
                    'id': email['id'],
                    'subject': email['subject'],
                    'sender': email['sender'],
                    'timestamp': email['timestamp'],
                    'categories': self.categorize_email(email)
                }
            )
            documents.append(doc)
        
        # Ensure documents are created
        if not documents:
            raise ValueError("No documents created for vector store index")
        
        # Create vector store index
        self.emails_index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.embed_model,  # Use the Hugging Face embedding model
            show_progress=True
        )
        print(f"Successfully indexed {len(documents)} emails")
    except Exception as e:
        print(f"Error initializing vector store: {e}")