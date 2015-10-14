       
if __name__ == '__main__':
    import csv
    import os
    from numpy import mean
    #module_path = os.path.dirname(__file__)
    #print module_path
    ## cores, accuracy, exeuction time
    spamReader = csv.reader(open('AnsonCores.csv', 'rb'), delimiter=',', quotechar='"')
    cores = {11:{}}
    for row in spamReader:
        cores[11][int(row[1])] = int(row[0])
    
    spamReader = csv.reader(open('AnsonPower.csv', 'rb'), delimiter=';', quotechar='"')

    allData = {}
    for row in spamReader:
        row_0 = int(row[0]) ##
        row_1 = int(row[1])
        row_2 = int(row[2])
        row_3 = int(row[3])
        data = float(row[4])
        
        try:
            try:
                try:
                    try:
                        if data != 0.0:
                            allData[row_0][row_1][row_2][row_3] = [data]
                    except:
                        allData[row_0][row_1][row_2][row_3].append(data)
                except:
                    allData[row_0][row_1][row_2] = {row_3:[data]}
            except:
                allData[row_0][row_1] = {row_2:{row_3:[data]}}
        except:
            allData[row_0] = {row_1:{row_2:{row_3:[data]}}}
    #print allData
    
    ### addd 23 - 4
   
    for we in range(11,54):
        for df in range(4,24):
            for core in range(1,cores[11][we]+1):
                try:
                    allData[11][we][core][df] = allData[11][we][core][24]
                except:
                    try:
                        print ":",we," ",df," ",core," ",
                        print allData[11][we][core].keys()
                    except:
                        print "cores:",we," ",df," ",core," ",
   ### check for holes
    print allData[11]
    ##write new spreadsheet
    with open('AnsonModPower.csv', 'wb') as csvfile:
        for key1, value1 in allData[11].iteritems(): 
            for key2, value2 in value1.iteritems(): 
                for key3, value3 in value2.iteritems(): 
                    spamwriter = csv.writer(csvfile,  delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow([11, key1, key2, key3, mean(value3)])