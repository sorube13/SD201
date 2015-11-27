from UserString import MutableString
import re
from collections import Counter

## It represents a document inside a text corpus.
class Document :
    def __init__(self, title, text, category = None) :
        self.title = title
        self.text = text
        self.category = category

    def get_category(self):
        return self.__category


    def set_category(self, value):
        self.__category = value


    def del_category(self):
        del self.__category

    
    def get_title(self) :
        return self.__title


    def get_text(self) :
        return self.__text


    def set_title(self, value) :
        self.__title = value


    def set_text(self, value) :
        self.__text = value


    def del_title(self) :
        del self.__title


    def del_text(self) :
        del self.__text
        
    def get_first_sentence(self) :
        dotIdx = self.text.find('.')
        if dotIdx != -1 :
            return self.text[:dotIdx]
        return self.text
        
    def __rep__(self):
        return "Title: " + self.title + ", Text: " + (self.text[:100] + "..." if len(self.text) > 100 else self.text )
    
    def __str__(self):
        return self.title + "\n" + self.category + "\n"  + self.text 
    
    '''
    It returns the number of words contained in the article's text. 
    '''
    def get_word_size(self):
        words = re.findall(r'\w+', self.text)
        ct = Counter(words)
        return sum(ct[word] for word in ct)
        

    title = property(get_title, set_title, del_title, "Document's title")
    text = property(get_text, set_text, del_text, "Document's content")
    category = property(get_category, set_category, del_category, "Document's real category")
    word_size = property(fget=get_word_size, fset=None, fdel=None, doc="Number of words in the document's content")

'''
Instance of this class allow to iterate through a corpus of documents stored 
sequentially in a text file.
The articles are stored as follows:
Title
Category label (optional)
Text
[Blank line]
Title
...
'''
class WikiCorpus :
    
    '''Class construct that takes a filehandler as argument.
    This must have been initialized: filehandler = open(filename)
    where filename is the name of the text file storing the corpus.
    labeled should be True if the corpus we are reading contains the 
    topic labels of the articles.'''
    def __init__(self, filehandler, labeled = False) :
        self.filehandler = filehandler
        self.labeled = labeled
    ## Cursor position
    currentTitle = ""
    currentCategory = ""
    currentText = MutableString()
        
    def __iter__(self) :
        while True :            
            nextLine = next(self.filehandler, None)
            if nextLine == None or nextLine == '\n' :
                if self.currentTitle != '' :
                    if self.labeled :
                        yield Document(str(self.currentTitle), str(self.currentText), str(self.currentCategory))
                    else :
                        yield Document(str(self.currentTitle), str(self.currentText))
                if nextLine == None :
                    break
                else :
                    self.currentTitle = ""
                    self.currentCategory = ""
                    self.currentText = MutableString()
            elif len(self.currentTitle) == 0 :
                self.currentTitle = nextLine.strip('\n')
            elif self.labeled and len(self.currentCategory) == 0 :
                self.currentCategory = nextLine.strip('\n')
            else :
                self.currentText.append(nextLine)

if __name__ == '__main__':
    with open('text') as file :                        
        corpus = WikiCorpus(file)
        
        for doc in corpus :
            print str(doc)
    