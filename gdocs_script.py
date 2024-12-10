from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle
import pprint as pp
import logging
from pathlib import Path
import json
from datetime import datetime

from app.utils import write_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDocsClient:
    """
    A client for interacting with Google Docs API using OAuth 2.0
    """
    
    # If modifying these scopes, delete the token.pickle file
    SCOPES = ['https://www.googleapis.com/auth/documents']
    
    def __init__(self, credentials_path: str = 'credentials.json'):
        """
        Initialize the Google Docs client
        
        Args:
            credentials_path: Path to the OAuth 2.0 credentials JSON file
        """
        self.credentials_path = credentials_path
        self.credentials = None
        self.service = None
    
    def authenticate(self):
        """
        Handle the OAuth 2.0 flow and authenticate the application
        """
        # Check if we have valid credentials saved
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.credentials = pickle.load(token)
        
        # If there are no valid credentials, let's get some
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                try:
                    self.credentials.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing credentials: {e}")
                    self.credentials = None
            
            if not self.credentials:
                try:
                    # Create flow instance with client secrets
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.SCOPES)
                    
                    # Run the OAuth flow
                    self.credentials = flow.run_local_server(port=8081)
                    
                    # Save the credentials for future use
                    with open('token.pickle', 'wb') as token:
                        pickle.dump(self.credentials, token)
                except Exception as e:
                    raise Exception(f"Failed to authenticate: {e}")
        
        # Build the service
        try:
            self.service = build('docs', 'v1', credentials=self.credentials)
            return self.service
        except Exception as e:
            raise Exception(f"Failed to build service: {e}")
    
    def get_document(self, document_id: str):
        """
        Retrieve a Google Doc by its ID
        
        Args:
            document_id: The ID of the Google Doc
            
        Returns:
            The document content
        """
        try:
            if not self.service:
                self.authenticate()
            
            # Call the Docs API
            if self.service:
                document = self.service.documents().get(documentId=document_id).execute()
                return document
        except Exception as e:
            raise Exception(f"Failed to get document: {e}")
    
    def read_paragraph_element(self, element):
        """
        Read the text content from a paragraph element
        
        Args:
            element: The paragraph element from the Google Doc
            
        Returns:
            The text content of the element
        """
        text_run = element.get('textRun')
        if not text_run:
            return ''
        return text_run.get('content', '')
    
    def read_structural_elements(self, elements):
        """
        Read text from structural elements
        
        Args:
            elements: The structural elements from the Google Doc
            
        Returns:
            The concatenated text content
        """
        text = []
        for element in elements:
            if 'paragraph' in element:
                elements = element.get('paragraph').get('elements')
                for elem in elements:
                    text.append(self.read_paragraph_element(elem))
            elif 'table' in element:
                # Handle tables if needed
                pass
            elif 'tableOfContents' in element:
                # Handle table of contents if needed
                pass
        return ''.join(text)

    def get_required_sections(self, elements):
        required_elements = []
        counter = 0
        start_stop_idx = []
        for element in elements:
            if 'paragraph' in element:
                elements = element.get('paragraph').get('elements')
                for el in elements:
                    if el.get('textRun').get('content').lower() in ['summary\n', 'education\n']:
                        start_stop_idx.append(counter)
            counter += 1
        print(start_stop_idx)




def main():
    """
    Example usage of the GoogleDocsClient
    """
    try:
        # Initialize the client
        client = GoogleDocsClient('credentials.json')
        
        # Authenticate and get service
        client.authenticate()
        
        # Example document ID (from the URL of a Google Doc)
        DOCUMENT_ID = '1g0IIso3qYjBxivTO5EhfGtc7S19f0J4U1Km7Kp5hEKY'
        
        # Get the document
        doc = client.get_document(DOCUMENT_ID)
        if doc:
            client.get_required_sections(doc.get('body').get('content'))
        # # Read and print the text content
        # if doc:
        #     doc_content = client.read_structural_elements(doc.get('body').get('content'))
        #     print(f"Document content:\n{doc_content}")
        # 
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
