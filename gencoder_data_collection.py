import urllib.request
import pandas as pd
import regex as re

df = pd.read_csv("<insert file path>")
accession_numbers = df["Code"].tolist()

def format(string):
  string = str(string)
  loc_split = string.find("genome")+6
  string = string[loc_split:]
  string = string.replace("\\n", "")
  return string

sequences = []

for accession_number in accession_numbers:
  url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id="+accession_number+".1&rettype=fasta&retmode=text"
  opener = urllib.request.build_opener()
  opener.addheaders = [('User-Agent', 'MyApp/1.0')]
  urllib.request.install_opener(opener)
  resp = urllib.request.urlopen(url)
  data = resp.read()
  sequence = format(data)
  sequences.append(sequence)

df["sequences"] = sequences
df.to_csv("<file name>", index = False)