from bratreader.repomodel import RepoModel

# load repomodel
r = RepoModel("/Users/sanatbek/code/uzabsa/ann-files")

doc = r.documents 			# get document with key 001
r.save_xml("xml-files")
