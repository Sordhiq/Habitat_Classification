import pandas as pd
import pickle
import gradio as gr
from Bio import Entrez, SeqIO
from Levenshtein import distance

with open("Classifier.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)


def habitat(species1, processid1, marker_code1, gb_acs1, nucraw1):
    
    """Habitat Classification Web App"""

    # User-defined values
    user_values = {
        'species': str(species1),
        'processid': str(processid1),
        'marker_code': str(marker_code1),
        'gb_acs': str(gb_acs1),
        'nucraw': str(nucraw1)
    }

    # Function to clean up the sequence strings
    def clean_sequence(seq):
        seq=str(seq)
        if seq.startswith('Seq(') and seq.endswith(')'):
            return seq[4:-1]
        return seq
    def clean_keys(key):
        return key.split('.')[0]

    def calculate_distance(row):
        nucraw = str(row['nucraw'])
        reference_seq = str(row['ref_nu'])
        return distance(nucraw, reference_seq)

    
    ##Fetching Reference sequence from ncbi sequentially
    
    # email address, which NCBI requires for access
    Entrez.email = "medhasharma3250@gmail.com"


    def fetch_sequence(accession_numbers):
        """
        Fetches nucleotide sequences from GenBank given a list of accession numbers.
        """
        try:
            # Post the list of accession numbers to NCBI
            search_results = Entrez.epost(db="nucleotide", id=",".join(accession_numbers))
            search_results_content = search_results.read().decode("utf-8")
            webenv = search_results_content.split("<WebEnv>")[1].split("</WebEnv>")[0]
            query_key = search_results_content.split("<QueryKey>")[1].split("</QueryKey>")[0]
    
            # Fetch the sequence data from NCBI
            handle = Entrez.efetch(db="nucleotide", rettype="gb", retmode="text", webenv=webenv, query_key=query_key)
            records = list(SeqIO.parse(handle, "genbank"))
            handle.close()
    
            return records
        except Exception as e:
            print(f"Error fetching data for accession numbers: {e}")
            return []
      
    # Example GenBank accession numbers
    accession_numbers = list(set(list(gb_acs1)))
    
    # Fetch the sequences in batch
    sequences_ = fetch_sequence(accession_numbers)
    
    # Dictionary to store fetched sequences
    sequences_dict_ = {record.id: record.seq for record in sequences_}
    #Clean sequences
    cleaned_sequence_dic_ = {clean_keys(k): clean_sequence(v) for k, v in sequences_dict_.items()}
    
    #Convert User values to df
    user_df=pd.DataFrame([user_values])
    
    #Add Reference DNA sequence column
    user_df['ref_nu']=user_df['gb_acs'].map(cleaned_sequence_dic_)
    
    
    #Calculate Levenshtein Distance
    
    user_df['levenshtein_distance'] = user_df.apply(calculate_distance, axis=1)
    
    ##Encoding values
    # Load the CSV file into a DataFrame
    encoding_df3 = pd.read_csv('encoders_sel_mapping.csv')
    
    # Create mapping dictionaries for each column
    mappings = {}
    for column in encoding_df3['Column'].unique():
        column_mappings = encoding_df3[encoding_df3['Column'] == column].set_index('Original Value')['Encoded Value'].to_dict()
        mappings[column] = column_mappings
    
    
    # Encode the object type columns in user_df
    for col in user_df.select_dtypes(include=['object']).columns:
        if col in mappings:
            user_df[col] = user_df[col].map(mappings[col])
    
    #Drop Ref_nu column
    user_df=user_df.drop('ref_nu', axis='columns')

    # Make the prediction
    predicted_encoded_class = classifier.predict(user_df)
    
    # Assuming the target encoding mapping is available in 'encoded_df3'
    # Reverse the mapping to get the original class
    target_column = 'habitat_type'  # Replace with the actual target column name
    reverse_mapping = {v: k for k, v in mappings[target_column].items()}

    predicted_class = reverse_mapping[predicted_encoded_class[0]]
        
    return predicted_class  # Return the predicted habitat


main = gr.Interface(
    fn=habitat,
    inputs=[
        gr.Textbox(label="Species"),
        gr.Textbox(label="Processid"),
        gr.Textbox(label="Marker Code"),
        gr.Textbox(label="GB_ACS"),
        gr.Textbox(label="Nucraw"),
    ],
    outputs=gr.Textbox(label="Predicted Habitat"),
    title="eDNA Habitat Classification",
    description="Welcome to the Habibat Classification Webapp as developed by SpaceM. This solution is aimed at understanding the underlying relationship between species, their nucleotide sequences, and how all of these influences their habitats."
)
main.launch()