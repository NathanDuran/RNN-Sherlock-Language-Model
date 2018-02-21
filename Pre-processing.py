import nltk;

fileInput = open("resources\The Adventures of Sherlock Holmes.txt" , "r");

print ("Name of the file: ", fileInput.name)
print ("Closed or not : ", fileInput.closed)
print ("Opening mode : ", fileInput.mode)