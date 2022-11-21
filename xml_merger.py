import glob
from xml.etree import ElementTree


def run(files):
    xml_files = glob.glob(files + "/*.xml")
    print(files)
    print(xml_files)
    xml_element_tree = None
    for xml_file in xml_files:
        # get root
        data = ElementTree.parse(xml_file).getroot()
        for result in data.iter('sentences'):
            if xml_element_tree is None:
                xml_element_tree = data
            else:
                xml_element_tree.extend(result)
    if xml_element_tree is not None:
        ElementTree.ElementTree(xml_element_tree).write('uzabsa-base-eval-valid/rest-manual-gold.xml')


run('data/xml-files')
