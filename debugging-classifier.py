import duckdb
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

class DebuggingConversationAnalyzer:
    def __init__(self, json_file_path):
        """
        Initialize the analyzer with the path to the conversation JSON file
        """
        self.json_file_path = json_file_path
        self.con = duckdb.connect(database=':memory:')
        self.debug_patterns = [
            r'error[s]?', r'bug[s]?', r'debug', r'fix', r'issue[s]?', r'problem[s]?',
            r'doesn\'t work', r'not working', r'traceback', r'exception',
            r'failed', r'undefined', r'null pointer', r'syntax error',
            r'runtime error', r'stacktrace', r'doesn\'t compile', r'won\'t compile',
            r'NaN', r'ValueError', r'TypeError', r'IndexError', r'KeyError',
            r'AttributeError', r'ImportError', r'SyntaxError'
        ]
        self.debug_keywords = [
            'debug', 'error', 'bug', 'issue', 'problem', 'exception',
            'traceback', 'fix', 'failed', 'failing', 'doesn\'t work',
            'broken', 'crash', 'troubleshoot', 'solve', 'incorrect'
        ]
        
    def load_data(self):
        """
        Load conversation data from JSON into DuckDB
        """
        print(f"Loading conversation data from {self.json_file_path}...")
        self.con.execute(f"""
            CREATE TABLE IF NOT EXISTS conversations AS 
            SELECT * FROM read_json_auto('{self.json_file_path}')
        """)
        
        # Extract conversation info
        conversation_count = self.con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        print(f"Loaded {conversation_count} conversations")
        
        # Create a flattened view of messages for easier analysis
        self.con.execute("""
            CREATE OR REPLACE VIEW messages AS
            WITH RECURSIVE message_list AS (
                SELECT 
                    uuid as conversation_id,
                    name as conversation_name, 
                    created_at as conversation_created_at,
                    UNNEST(chat_messages) as message 
                FROM conversations
            )
            SELECT 
                conversation_id,
                conversation_name,
                conversation_created_at,
                message['sender'] as sender,
                message['text'] as text,
                message['created_at'] as created_at
            FROM message_list
            WHERE text IS NOT NULL AND text != ''
        """)

    def identify_debugging_conversations(self):
        """
        Identify conversations that are likely to be about debugging
        """
        pattern_str = '|'.join(self.debug_patterns)
        
        # First pass: identify conversations with debugging-related terms
        escaped_pattern = pattern_str.replace("'", "''")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE debug_conversations AS
            SELECT 
                c.uuid as conversation_id,
                c.name as conversation_name,
                c.created_at as created_at,
                ARRAY_LENGTH(c.chat_messages) as message_count,
                COUNT(CASE WHEN REGEXP_MATCHES(LOWER(m.text), '{escaped_pattern}') THEN 1 END) as debug_message_count
            FROM conversations c
            JOIN messages m ON c.uuid = m.conversation_id
            GROUP BY c.uuid, c.name, c.created_at, c.chat_messages
            HAVING COUNT(CASE WHEN REGEXP_MATCHES(LOWER(m.text), '{escaped_pattern}') THEN 1 END) > 0
            ORDER BY debug_message_count DESC
        """)
        
        # Calculate debug intensity (% of messages with debug terms)
        self.con.execute("""
            ALTER TABLE debug_conversations 
            ADD COLUMN debug_intensity DOUBLE;
            
            UPDATE debug_conversations 
            SET debug_intensity = debug_message_count::DOUBLE / message_count::DOUBLE
        """)
        
        debug_count = self.con.execute("SELECT COUNT(*) FROM debug_conversations").fetchone()[0]
        total_count = self.con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        percent = (debug_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"Identified {debug_count} potential debugging conversations ({percent:.2f}% of total)")
        return self.con.execute("SELECT * FROM debug_conversations ORDER BY debug_intensity DESC").fetchdf()
    
    def extract_conversation_features(self):
        """
        Extract features for classifier training
        """
        # Get all conversations with their full text
        conversations_df = self.con.execute("""
            SELECT 
                c.uuid as conversation_id,
                c.name as conversation_name,
                STRING_AGG(m.text, ' ') as full_text
            FROM conversations c
            JOIN messages m ON c.uuid = m.conversation_id
            GROUP BY c.uuid, c.name
        """).fetchdf()
        
        # Label conversations that match debug patterns
        pattern = '|'.join(self.debug_patterns)
        conversations_df['is_debug'] = conversations_df['full_text'].str.lower().str.contains(pattern).astype(int)
        
        # Create feature matrix using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000, 
            min_df=2, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(conversations_df['full_text'])
        y = conversations_df['is_debug']
        
        # Get feature names for later interpretation
        feature_names = vectorizer.get_feature_names_out()
        
        return conversations_df, X, y, feature_names, vectorizer
    
    def train_classifier(self, X, y):
        """
        Train a debugging conversation classifier
        """
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = classifier.predict(X_test)
        print("\nClassifier performance:")
        print(classification_report(y_test, y_pred))
        
        return classifier, X_test, y_test
    
    def analyze_debug_conversations(self, classifier, vectorizer):
        """
        Analyze debugging conversations to extract insights
        Using simpler methods to avoid transformer model issues
        """
        # Get the full content of debug conversations
        debug_conversations = self.con.execute("""
            SELECT 
                c.uuid as conversation_id,
                c.name as conversation_name,
                c.created_at,
                ARRAY_AGG(STRUCT_PACK(
                    sender := m.sender, 
                    text := m.text,
                    created_at := m.created_at
                ) ORDER BY m.created_at) as messages
            FROM conversations c
            JOIN debug_conversations dc ON c.uuid = dc.conversation_id
            JOIN messages m ON c.uuid = m.conversation_id
            GROUP BY c.uuid, c.name, c.created_at
        """).fetchdf()
        
        # Initialize sentiment analysis pipeline with truncation enabled to handle long texts
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )
        
        # Initialize NLI model for success/failure classification with truncation
        success_classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli",
            truncation=True,
            max_length=512
        )
        
        # Results storage
        results = []
        
        # Simple debug patterns for resolution detection
        resolved_patterns = [
            r'(thank you|thanks|worked|resolved|fixed|solved|success|great|awesome|excellent)',
            r'(it works|problem solved|issue fixed|bug fixed|working now)',
            r'(appreciate|helpful|perfect|exactly what I needed)'
        ]
        
        # Define patterns for errors
        error_pattern = r'(?:error|exception|traceback)[\s\:]+(.*?)(?:\n\n|\.\s|$)'
        
        # Process each conversation
        for idx, row in debug_conversations.iterrows():
            print(f"Analyzing conversation {idx+1} of {len(debug_conversations)}: {row['conversation_name']}")
            
            # Extract messages
            messages = row['messages']
            
            # Reconstruct conversation flow
            convo_text = ""
            human_frustration = []
            assistant_helpfulness = []
            message_sequence = []
            human_messages = []
            assistant_messages = []
            
            # Process messages chronologically
            for msg in messages:
                sender = msg['sender']
                text = msg['text']
                convo_text += f"{text} "
                
                # Store messages by role
                if sender == 'human':
                    human_messages.append(text)
                else:
                    assistant_messages.append(text)
                
                message_sequence.append({
                    'role': sender,
                    'text': text[:100] + "..." if len(text) > 100 else text
                })
                
                # Analyze sentiment of human messages - only if text isn't too long to prevent warning
                if sender == 'human' and len(text.split()) > 3:
                    sentiment = sentiment_analyzer(text[:512] if len(text) > 512 else text)[0]
                    human_frustration.append(sentiment)
                
                # Analyze perceived helpfulness of assistant messages
                if sender == 'assistant' and len(text.split()) > 3:
                    sentiment = sentiment_analyzer(text[:512] if len(text) > 512 else text)[0]
                    assistant_helpfulness.append(sentiment)
            
            # Determine if bug was resolved using zero-shot classification
            success_labels = ["problem solved", "issue resolved", "debugging successful", 
                             "found solution", "fixed bug"]
            failure_labels = ["problem persists", "issue unresolved", "debugging failed", 
                             "no solution", "bug remains"]
            
            # Combine both sets of labels
            all_labels = success_labels + failure_labels
            
            # Run classification on a truncated version of the conversation text to avoid exceeding model limits
            # Truncate to 1000 characters to fit within the model's limit after tokenization
            truncated_convo_text = convo_text[:1000] if len(convo_text) > 1000 else convo_text
            outcome_classification = success_classifier(truncated_convo_text, all_labels)
            
            # Check if any success label has higher score than all failure labels
            success_scores = [score for label, score in zip(outcome_classification['labels'], 
                                                         outcome_classification['scores']) 
                           if label in success_labels]
            failure_scores = [score for label, score in zip(outcome_classification['labels'], 
                                                         outcome_classification['scores']) 
                           if label in failure_labels]
            
            if max(success_scores) > max(failure_scores):
                resolution_outcome = "Likely Resolved"
                resolution_confidence = max(success_scores)
            else:
                resolution_outcome = "Likely Unresolved"
                resolution_confidence = max(failure_scores)
            
            # Calculate metrics
            avg_human_sentiment = np.mean([h['score'] if h['label'] == 'POSITIVE' else -h['score'] 
                                        for h in human_frustration]) if human_frustration else 0
            avg_assistant_helpfulness = np.mean([a['score'] if a['label'] == 'POSITIVE' else -a['score']
                                              for a in assistant_helpfulness]) if assistant_helpfulness else 0
            
            # Extract key debugging info using pattern matching
            error_pattern = r'(?:error|exception|traceback)[\s\:]+(.*?)(?:\n\n|\.\s|$)'
            errors_found = re.findall(error_pattern, convo_text, re.IGNORECASE)
            errors_text = "; ".join(errors_found[:3])  # Limit to first 3 errors
            
            # Store results
            results.append({
                'conversation_id': row['conversation_id'],
                'conversation_name': row['conversation_name'],
                'created_at': row['created_at'],
                'message_count': len(messages),
                'human_sentiment': avg_human_sentiment,
                'assistant_helpfulness': avg_assistant_helpfulness,
                'resolution_outcome': resolution_outcome,
                'resolution_confidence': resolution_confidence,
                'errors_identified': errors_text[:200] if errors_text else "No specific errors extracted",
                'message_sample': message_sequence[:3]  # Sample of first few messages
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        return results_df
    
    def visualize_results(self, results_df):
        """
        Create visualizations of debugging analysis results
        """
        plt.figure(figsize=(10, 6))
        
        # Plot resolution outcomes
        sns.countplot(x='resolution_outcome', data=results_df)
        plt.title('Debugging Resolution Outcomes')
        plt.tight_layout()
        plt.savefig('debug_resolution_outcomes.png')
        
        # Plot sentiment vs helpfulness
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='human_sentiment', y='assistant_helpfulness', 
                       hue='resolution_outcome', data=results_df)
        plt.title('Human Sentiment vs Assistant Helpfulness')
        plt.xlabel('Human Sentiment (Negative to Positive)')
        plt.ylabel('Assistant Helpfulness (Negative to Positive)')
        plt.tight_layout()
        plt.savefig('sentiment_helpfulness.png')
        
        # Distribution of message counts in debugging conversations
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_df, x='message_count', bins=20)
        plt.title('Message Count Distribution in Debugging Conversations')
        plt.xlabel('Number of Messages')
        plt.tight_layout()
        plt.savefig('debug_message_counts.png')
        
        return "Visualizations saved as PNG files"

def main(json_file_path):
    """
    Run the complete debugging conversation analysis pipeline
    """
    analyzer = DebuggingConversationAnalyzer(json_file_path)
    analyzer.load_data()
    
    # Identify potential debugging conversations
    debug_convos = analyzer.identify_debugging_conversations()
    print(f"Top 5 debugging conversations by debug intensity:")
    print(debug_convos.head(5)[['conversation_name', 'debug_message_count', 'debug_intensity']])
    
    # Extract features and train classifier
    print("\nExtracting features and training classifier...")
    conversations_df, X, y, feature_names, vectorizer = analyzer.extract_conversation_features()
    classifier, X_test, y_test = analyzer.train_classifier(X, y)
    
    # Analyze the debugging conversations in detail
    print("\nAnalyzing debugging conversations in detail...")
    analysis_results = analyzer.analyze_debug_conversations(classifier, vectorizer)
    
    # Display summary of results
    print("\nDebugging Analysis Summary:")
    print(f"Total debugging conversations analyzed: {len(analysis_results)}")
    resolution_counts = analysis_results['resolution_outcome'].value_counts()
    for outcome, count in resolution_counts.items():
        print(f"  {outcome}: {count} ({count/len(analysis_results)*100:.1f}%)")
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.visualize_results(analysis_results)
    
    # Return dataframe for further analysis
    return analysis_results

if __name__ == "__main__":
    json_file_path = "conversations.json"
    results = main(json_file_path)
    
    # Export results to CSV
    results.to_csv("debugging_analysis_results.csv", index=False)
    print("\nResults exported to debugging_analysis_results.csv")
