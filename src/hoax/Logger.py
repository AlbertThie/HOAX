import tables as tb

class Logger():
    def __init__(self,loggingFile,epochs,epochStep):
        self.epochSize = epochs/epochStep
        #self.NeuralNetworkRun.setValidationError(epochs/epochStep)
        NeuralNetworkRun = {
                "idNumber" : tb.Col.from_type("int64"),
                "hiddenLayers" : tb.Col.from_type("int64"),
                "nodesPerLayer" : tb.Col.from_type("int64"),
                "batchsize" : tb.Col.from_type("int64"),
                "learningRate" : tb.Col.from_type("float64"),
                "validationError" : tb.Col.from_type("float64",(epochs/epochStep,)),   
                    }
        self.loggingFile = loggingFile
        self.logFile = tb.open_file(self.loggingFile,mode="w")
        self.root = self.logFile.root
        self.group = self.logFile.create_group(self.root,"NeuralNetworkRun")
        self.gRuns = self.root.NeuralNetworkRun 
        self.idnumber = 0
        table = self.logFile.create_table("/NeuralNetworkRun","NeuralNetworkRun1",NeuralNetworkRun,"Runs:"+"NeuralNetworkRun1")
        table.flush()
        print("writng file")
        print(self.group)
        print(self.root)
        self.logFile.close()



    def getLogFile(self):
        return tb.open_file(self.loggingFile,mode="a")

    def writeRun(self,hiddenlayer_number,hiddenlayer_size,batch_size,learning_rate,printinglog):
        filesaving = self.getLogFile()
        root = filesaving.root
        addTable=table.row 
        addTable['idNumber'] = self.idnumber
        addTable['hiddenLayers'] = hiddenlayer_number
        addTable['nodesPerLayer'] = hiddenlayer_size
        addTable['batchsize'] = batch_size
        addTable['learningRate'] = learning_rate
        addTable['validationError'] =np.array(printinglog)
        addTable.append()
        table.flush()
        filesaving.close()
        self.idnumber += 1