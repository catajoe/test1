from random import random
import math
import matplotlib.pyplot as plt

def creaDataset():

    rows=[]
    with open('Dataset.data', 'r') as f:
          data = f.readlines()
    
          for line in data:
              words = line.split()
              
              
              # una serie di cast
              
              lread= float(words[1])
              lwrite= float(words[2])
              scall= float(words[3])
              sread= float(words[4])
              swrite= float(words[5])   
              fork= float(words[6])      
              exes= float(words[7])      
              rchar= float(words[8])     
              wchar= float(words[9])     
              pgout= float(words[10])    
              ppgout= float(words[11])    
              pgfree= float(words[12])    
              pgscan= float(words[13])    
              atch= float(words[14])      
              pgin= float(words[15])      
              ppgin= float(words[16])     
              pflt= float(words[17])      
              vflt= float(words[18])      
              runqsz=float(words[19])    
              runocc= float(words[20])    
              freemem= float(words[21])   
              freeswap= float(words[22])
              cpu= float(words[23])
              
              # Add to the dataset
              rows.append({'input':(lread,lwrite,scall,sread,swrite,fork,exes,rchar,
                                    wchar,pgout,ppgout,pgfree,pgscan,atch,
                                    pgin,ppgin,pflt,vflt,runqsz,runocc,freemem,freeswap),
                     'result':cpu})


    return rows
        
def euclidean(v1,v2):
  d=0.0
  for i in range(len(v1)):
    d+=(v1[i]-v2[i])**2
  return math.sqrt(d)
  
def getdistances(data,vec1):
  distancelist=[]
  
  # Loop over every item in the dataset
  for i in range(len(data)):
    vec2=data[i]['input']
    
    # Add the distance and the index
    distancelist.append((euclidean(vec1,vec2),i))
  
  # Sort by distance
  distancelist.sort()
  return distancelist

def knnestimate(data,vec1,k):
  # Get sorted distances
  dlist=getdistances(data,vec1)
  avg=0.0
  
  # Take the average of the top k results
  for i in range(k):
    idx=dlist[i][1]
    avg+=data[idx]['result']
  avg=avg/k
  return avg  
  
  
def gaussian(dist,sigma=5.0):
  return math.e**(-dist**2/(2*sigma**2))  

def weightedknn(data,vec1,k=5,weightf=gaussian):
  # Get distances
  dlist=getdistances(data,vec1)
  avg=0.0
  totalweight=0.0
  
  # Get weighted average
  for i in range(k):
    dist=dlist[i][0]
    idx=dlist[i][1]
    weight=weightf(dist)
    avg+=weight*data[idx]['result']
    totalweight+=weight
  if totalweight==0: return 0
  avg=avg/totalweight
  return avg  
  
  
  
#-----------CROSS VALIDATION-------------

def dividedata(data,test=0.05):
  trainset=[]
  testset=[]
  for row in data:
    if random()<test:
      testset.append(row)
    else:
      trainset.append(row)
  return trainset,testset

def testalgorithm(algf,trainset,testset):
  error=0.0
  for row in testset:
    guess=algf       #passata avg
    error+=(row['result']-guess)**2 
    #print row['result'],guess
  #print error/len(testset)
  return error/len(testset)

def crossvalidate(algf,data,trials=100,test=0.5):
  error=0.0
  for i in range(trials):
    trainset,testset=dividedata(data,test)
    error+=testalgorithm(algf,trainset,testset)
  return error/trials  
  
  
  
  # Un ottimo main
  
def main():
    
    # Abbiamo memorizzato il nostro dataset in un dizionario
    vector= creaDataset()
    print "\nStampiamo il primo elemento del dizionario per prova: \n"
    print vector[0]
    
    d=(44,0,3700,410,200,3.39,5.79,544367,247202,2.99,7.98,21.16,51.10,1.20,21.36,53.9,618.16,427.15,2.0,20,180,1118221)
    print "\nQuesto e' il nostro nuovo item (per semplicita assumiamo il primo elemento del dataset): \n"
    print d
    
    k = int(raw_input("Introduci il numero di vicini: "))
    kA = k
    
    print "\n-------------------------------------------------\n"
    
    avg= knnestimate(vector,d,k)                     
    
    print "\nValore della predizione valutato attraverso il 'knnestimate'in percentuale: \n"
    print avg
    
    
    
    avg_w= weightedknn(vector,d,k)                     
    
    print "\nValore della predizione valutato attraverso il 'weightedknn'in percentuale: \n"
    print avg_w
    
    
    print "\nUtilizziamo la cross validation con k inserito dall'utente e utilizzando il 'knnestimate': \n"
   
    cross11= crossvalidate(avg,vector)
    print cross11
    
    print "\nUtilizziamo la cross validation con k inserito dall'utente e utilizzando il 'weightedknn': \n"
    cross12= crossvalidate(avg_w,vector)
    print cross12
    
    
    
    print "\n\nConfrontiamo la stessa funzione con un numero di vicini pari a 1\n"
    avg2= knnestimate(vector,d,k=1)
    cross21= crossvalidate(avg2,vector)
    print cross21
    print "\ncon K-nn ponderato: \n"
    avg_w2= weightedknn(vector,d,k=1)                     
    cross22= crossvalidate(avg_w2,vector)
    print cross22
    
    print "\n\nConfrontiamo la stessa funzione con un numero di vicini pari a 6\n"
    avg3= knnestimate(vector,d,k=6)
    cross31= crossvalidate(avg3,vector)
    print cross31
    print "\ncon K-nn ponderato: \n"
    avg_w3= weightedknn(vector,d,k=6)                     
    cross32= crossvalidate(avg_w3,vector)
    print cross32
    
    print "\n\nConfrontiamo la stessa funzione con un numero di vicini pari a 10\n"
    avg4= knnestimate(vector,d,k=10)
    cross41= crossvalidate(avg4,vector)
    print cross41
    print "\ncon K-nn ponderato: \n"
    avg_w4= weightedknn(vector,d,k=10)                     
    cross42= crossvalidate(avg_w4,vector)
    print cross42
    
    #---------------------grafico-------------------------------------
    
    # impostazione font assi
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    
    
     
    # creazione canvas (le misure sono in pollici))
    plt.figure(figsize=(8, 4.5))
     
    # disegno grafico knn stimate
    red,=plt.plot([1,6,10,kA], [cross21,cross31,cross41,cross11],'ro')
    
    #knn ponderato
    blue,=plt.plot([1,6,10,kA], [cross22,cross32,cross42,cross12],'bo')
    
    #range
    if(kA>10):
        plt.axis([0,kA+5,200,500])
    else:
        plt.axis([0,12,200,500])
    
    #Legend
    plt.legend([red, blue], ["K-nn", "K-nn ponderato"])
    

    # titolo grafico
    plt.title("Confronto tra k-nn e k-nn ponderato")
     
    #Etichette
    plt.xlabel('Numero di vicini')
    plt.ylabel('Cross Validate')
    
    #Griglia
    plt.grid(True) 
    
     
    # salvataggio su file in png
    plt.savefig("grafico.png")



if __name__ == "__main__":
    main()