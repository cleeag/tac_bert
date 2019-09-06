1. wiki-title-yago-types.txt : page id to types
2. enwiki-20190101-anchor-mentions-notok.txt : every mention. use page id to link to types, use wid + sent id to link to sentences
the key file. use this file to access everything
3. enwiki-20190101-anchor-sents-notok.txt : wid and sent id to sentence

## first task:
get type count, filter out types with the more mentions.
get men-type dict, and type sets. iterate through type set, count mention times, get type-mention time dict

1. pageID2type.pkl
    - pageID to types, dictionary
    - e.g. 49860890 : \['! (album)', 'Object100002684;Medium106254669;Album106591815;Whole100003553;Artifact100021939;PhysicalEntity100001930;Instrumentality103575240\n']

2. mention2type_dict.pkl
    - mention string to \[title, types], dictionary
    - e.g. self-governed : \['Federacy', 'SocialGroup107950920;Abstraction100002137;Group100031264;PoliticalSystem108367880\n']

3. type_count_dict.pkl
