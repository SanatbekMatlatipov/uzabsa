from bratreader.repomodel import RepoModel

# load repomodel, change this directory according to your need
r = RepoModel("/Users/sanatbek/code/uzabsa/data/manual_evaluation_gold")

doc = r.documents 			# get document with key 001
r.save_xml("data/xml-files")
