import csv

def txt_to_csv(txt_file, csv_file):
    """
    Convert a TXT file to CSV format, handling multiple spaces as delimiters.
    
    Args:
        txt_file (str): Path to the input TXT file.
        csv_file (str): Path to the output CSV file.
    """
    with open(txt_file, 'r', encoding='utf-8') as infile, open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        
        for line in infile:
            row = line.strip().split() 
            csv_writer.writerow(row)

txt_to_csv('list_landmarks_celeba.txt', 'data.csv')
 