#import feat_select as fs
#import pipeline as pp
#import train_ind as ti
#import train_comb as tc
import sys
import tkinter as tk
from tkinter import filedialog
import random
import webbrowser

class GUI:

    def __init__(self, master):

        self.master = master
        master.title("Multi Modal Genomics - Breast Cancer Survival Prediction")

        frame1 = tk.Frame(master, bd=0, highlightbackground="black", highlightcolor="black",
                          highlightthickness=1, pady=5)
        frame1.pack(side="top", anchor=tk.NW)
        frame2 = tk.Frame(master, bd=0, highlightbackground="black", highlightcolor="black",
                          highlightthickness=1, pady=5)
        frame2.pack(side="bottom")

        label1 = tk.Label(frame1, text="Optional inputs for integrated classification:", background="white")
        label1.pack(anchor=tk.NW)

        frameModalities = tk.Frame(frame1)
        frameModalities.pack(anchor=tk.N,side="left",padx=20)
        labelModalities = tk.Label(frameModalities,
                                   text="Select which method to use for\nfeature selection of  variables:")
        labelModalities.pack(anchor=tk.W)

        var_t = tk.IntVar()
        checkbox_t = tk.Checkbutton(frameModalities, text="T test (cont) / Chi-square (disc)", variable=var_t)
        checkbox_t.pack(anchor=tk.W)
        var_mRMR = tk.IntVar()
        checkbox_mRMR = tk.Checkbutton(frameModalities, text="mRMR", variable=var_mRMR)
        checkbox_mRMR.pack(anchor=tk.W)

        frameRem = tk.Frame(frame1)
        frameRem.pack(anchor=tk.N, side="left", padx=20)
        labelRem = tk.Label(frameRem, text="Check ""remove"" below to remove non-coding\nRNA "
                                           "sequences from RNA data sets:")
        labelRem.pack(anchor=tk.N)

        var_rem = tk.IntVar()
        checkbox_rem = tk.Checkbutton(frameRem, text="Remove", variable=var_rem)
        checkbox_rem.pack(anchor=tk.W)


        frameRand = tk.Frame(frame1)
        frameRand.pack(anchor=tk.N, side="left", padx=20)
        labelRand = tk.Label(frameRand, text="Select a whole number to be used as a seed"
                                                  "\nfor random number generation:")
        labelRand.pack(anchor=tk.N)

        var_Rand = tk.IntVar()
        var_Rand = 5
        entry_Rand = tk.Entry(frameRand, textvariable=str(var_Rand))
        entry_Rand.insert("end", str(var_Rand))
        def rnd():
            var_Rand = random.randint(1,50)
            entry_Rand.delete(0,tk.END)
            entry_Rand.insert("end", str(var_Rand))
        button_Rand = tk.Button(frameRand, text="Random", foreground="black", bd=2,
                                     command=rnd, height=2, width=10)
        button_Rand.pack()

        entry_Rand.pack()

        frameTrain = tk.Frame(frame1)
        frameTrain.pack(anchor=tk.N, side="left", padx=20)
        labelTrain = tk.Label(frameTrain, text="Select percentage of total patients to be "
                                                     "\nincluded in the test group for classification:")
        labelTrain.pack(anchor=tk.N)

        var_Train = tk.StringVar()
        entry_Train = tk.Entry(frameTrain, textvariable=var_Train)
        entry_Train.insert("end", "0.15")
        entry_Train.pack()
        ## MAKE BOX BELOW TO SHOW HOW MANY WILL BE IN EACH GROUP (ceil for rounding)

        frame3 = tk.Frame(frame2, bd=0, highlightbackground="black", highlightcolor="black",
                          highlightthickness=1, pady=5)
        frame3.pack(side="left")
        frame4 = tk.Frame(frame2, bd=0, highlightbackground="black", highlightcolor="black",
                          highlightthickness=1, pady=5)
        frame4.pack(side="right")

        label2 = tk.Label(frame3, text="Classification Results:", background="white")
        label2.pack(side="top", anchor=tk.NW)
        frameOutputClassifier = tk.Frame(frame3)
        frameOutputClassifier.pack(anchor=tk.NW, side="top", padx=5, pady=5)
        labelOutputClassifier = tk.Label(frameOutputClassifier, text="Classifier Output:")
        labelOutputClassifier.pack(anchor=tk.NW)
        scrollbar_out = tk.Scrollbar(frameOutputClassifier)
        scrollbar_out.pack(side="right", fill=tk.Y)
        text_OutputClassifier = tk.Text(frameOutputClassifier, state=tk.DISABLED, height=5, width=80,
                                        yscrollcommand=scrollbar_out.set)
        text_OutputClassifier.pack(expand=True, fill='both')  # check how to be able to show more

        frameOutputFeatures = tk.Frame(frame3)
        frameOutputFeatures.pack(anchor=tk.SW, side="bottom", padx=5, pady=5)

        frameOutputGene = tk.Frame(frameOutputFeatures)
        frameOutputGene.pack(anchor=tk.SW, side="left", padx=5, pady=5)
        labelOutputGene = tk.Label(frameOutputGene, text="Selected Features:\nGene Expression")
        labelOutputGene.pack(side="top", anchor=tk.SW)

        scrollbar_outGene = tk.Scrollbar(frameOutputGene)
        scrollbar_outGene.pack(side="right", fill=tk.Y)
        text_OutputGene = tk.Text(frameOutputGene, state=tk.DISABLED, height=5, width=20,
                                      yscrollcommand=scrollbar_outGene.set)
        text_OutputGene.pack(expand=True, fill='both')

        def openGeneFeats():
            windowGene = tk.Toplevel(root)
            frame = tk.Frame(windowGene, bd=0, highlightbackground="black", highlightcolor="black",
                              highlightthickness=1, pady=5)
            frame.pack(side="top", anchor=tk.NW)
            hyperlink = tk.Label(frame, text="Gene database hyperlink", fg="blue", cursor="hand2")
            hyperlink.pack()
            def click(event):
                url = 'https://useast.ensembl.org/index.html'
                webbrowser.open_new(url)
            hyperlink.bind("<Button-1>", click)
            frame2 = tk.Frame(windowGene, bd=0, highlightbackground="black", highlightcolor="black",
                              highlightthickness=1, pady=5)
            frame2.pack(side="bottom")
            #text = features
        button_OutputGene = tk.Button(frameOutputGene, text="Gene Features", foreground="black", bd=2,
                                     command=openGeneFeats, height=1, width=18)
        button_OutputGene.pack()

        frameOutputmi = tk.Frame(frameOutputFeatures)
        frameOutputmi.pack(anchor=tk.SW, side="left", padx=5, pady=5)
        labelOutputmi = tk.Label(frameOutputmi, text="Selected Features:\nmiRNA Expression")
        labelOutputmi.pack(side="top", anchor=tk.SW)

        scrollbar_outmi = tk.Scrollbar(frameOutputmi)
        scrollbar_outmi.pack(side="right", fill=tk.Y)
        text_Outputmi = tk.Text(frameOutputmi, state=tk.DISABLED, height=5, width=20,
                                  yscrollcommand=scrollbar_outmi.set)
        text_Outputmi.pack(expand=True, fill='both')
        def openmiFeats():
            windowmi = tk.Toplevel(root)
        button_Outputmi = tk.Button(frameOutputmi, text="miRNA Features", foreground="black", bd=2,
                                     command=openmiFeats, height=1, width=18)
        button_Outputmi.pack()

        frameOutputMeth = tk.Frame(frameOutputFeatures)
        frameOutputMeth.pack(anchor=tk.SW, side="left", padx=5, pady=5)
        labelOutputMeth = tk.Label(frameOutputMeth, text="Selected Features:\nDNA Methylation")
        labelOutputMeth.pack(side="top", anchor=tk.SW)

        scrollbar_outMeth = tk.Scrollbar(frameOutputMeth)
        scrollbar_outMeth.pack(side="right", fill=tk.Y)
        text_OutputMeth = tk.Text(frameOutputMeth, state=tk.DISABLED, height=5, width=20,
                                yscrollcommand=scrollbar_outMeth.set)
        text_OutputMeth.pack(expand=True, fill='both')

        def openMethFeats():
            windowMeth = tk.Toplevel(root)

        button_OutputMeth = tk.Button(frameOutputMeth, text="miRNA Features", foreground="black", bd=2,
                                    command=openMethFeats, height=1, width=18)
        button_OutputMeth.pack()

        frameOutputCNV = tk.Frame(frameOutputFeatures)
        frameOutputCNV.pack(anchor=tk.SW, side="left", padx=5, pady=5)
        labelOutputCNV = tk.Label(frameOutputCNV, text="Selected Features:\nCopy Number Variation")
        labelOutputCNV.pack(side="top", anchor=tk.SW)

        scrollbar_outCNV = tk.Scrollbar(frameOutputCNV)
        scrollbar_outCNV.pack(side="right", fill=tk.Y)
        text_OutputCNV = tk.Text(frameOutputCNV, state=tk.DISABLED, height=5, width=20,
                                yscrollcommand=scrollbar_outCNV.set)
        text_OutputCNV.pack(expand=True, fill='both')

        def openCNVFeats():
            windowCNV = tk.Toplevel(root)

        button_OutputCNV = tk.Button(frameOutputCNV, text="CNV Features", foreground="black", bd=2,
                                    command=openCNVFeats, height=1, width=18)
        button_OutputCNV.pack()

        def runClassification():
            oneCheck = sum([var_t.get(), var_mRMR.get()])
            featSelectMethod = ''
            if oneCheck > 1:
                print("Only select one method for feature selection")
                sys.exit()
            if oneCheck == 1:
                if var_t.get() == 1:
                    featSelectMethod = 'ttest'
                else:
                    featSelectMethod = 'mrmr'
            else:
                print("Please select a method for feature selection")
                sys.exit()


        button_run = tk.Button(frame1, text="RUN", foreground="white", bg="green", bd=2,
                                     command=runClassification, height=4, width=20)
        button_run.pack_propagate(0)
        button_run.pack(anchor=tk.N, side="right")

        frameTestPt = tk.Frame(frame4)
        frameTestPt.pack(anchor=tk.N, side="top", padx=20, pady=20)
        labelTestPt = tk.Label(frameTestPt, text="Upload test patient data:")
        labelTestPt.pack(side="top", anchor=tk.NW)
        #
        # scrollbar_testPt = tk.Scrollbar(frameTestPt)
        # scrollbar_testPt.pack(side="right", fill=tk.Y)
        # listboxTestPt = tk.Listbox(frameTestPt, bg="white", height=8, width=20, yscrollcommand=scrollbar_testPt.set)
        # listboxTestPt.pack()

        def askOpenFile():
            file = filedialog.askopenfilename()
            if file != None:
                data = file.read()
                file.close()
                print("I got %d bytes from this file." % len(data))

        buttonUploadPt = tk.Button(frameTestPt, text="Browse for a patient data set", bg="yellow",
                                   command=askOpenFile)
        buttonUploadPt.pack(side="bottom")
        # Need to add what to do with file

        frameTestOutClass = tk.Frame(frame4)
        frameTestOutClass.pack(anchor=tk.N, side="left", padx=20, pady=20)
        labelTestOutClass = tk.Label(frameTestOutClass, text="Selected patient survival prediction: ")
        labelTestOutClass.pack(anchor=tk.NW)

        text_TestClass = tk.Text(frameTestOutClass, state=tk.DISABLED, height=5, width=15)
        text_TestClass.pack(expand=True, fill='both')  # check how to be able to show more

        frameTestOutProb = tk.Frame(frame4)
        frameTestOutProb.pack(anchor=tk.N, side="right", padx=20, pady=20)
        labelTestOutProb = tk.Label(frameTestOutProb, text="Survival prediction Probability: ")
        labelTestOutProb.pack(anchor=tk.NW)

        text_TestProb = tk.Text(frameTestOutProb, state=tk.DISABLED, height=5, width=15)
        text_TestProb.pack(expand=True, fill='both')  # check how to be able to show more

root = tk.Tk()
gui = GUI(root)

root.mainloop()
