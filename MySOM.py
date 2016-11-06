from __future__ import division
import itertools
import collections
import numpy as np
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
from math import *
import math
import sys
import scipy
from random import *
from PIL import ImageFont
from PIL import ImageDraw 

class SOM:
    def __init__(self, file_names, learning_rate = 0.001):
        self.kmer = 1
        self.windowed = True
        self.windowsize = 10000
        self.boolean = True
        self.iterations = 5
        self.batch = True
        self.radiusfactor = 3
        [fv, self.trainHeaders] = self.generate_dataVectors(file_names)
        fv2 = np.array(fv)
        
        dataMatrix = fv2[:,2:]   # Convert a list-of-lists into a numpy array.  aListOfLists is the data points in a regular list-of-lists type matrix.

        
        myPCA = PCA(dataMatrix)     # make a new PCA object from a numpy array object
        x_av = dataMatrix.mean(0)
        eigenvector_1 = myPCA.Wt[0]
        eigenvector_2 = myPCA.Wt[1]
        std1 = np.std(myPCA.Y[:, 0]) #calculating the standard deviation accross the first PC
        std2 = np.std(myPCA.Y[:, 1]) #calculating the standard deviation accross the second PC
        SOM_width = int(math.ceil(5*std1))
        SOM_height = int(math.ceil((std2/std1) * SOM_width))
        
        self.width = SOM_width
        self.height = SOM_height
        self.radius = max(self.height, self.width)/self.radiusfactor
        self.learning_rate = learning_rate
        self.FV_size = len(dataMatrix[0])
        self.trainV = fv2
        
        wt = scipy.array([[[0.0 for i in range(self.FV_size)] for x in range(self.width)] for y in range(self.height)])
        for i in range(SOM_height):
            for j in range(SOM_width):
                wt[i,j] = (x_av + ((eigenvector_1*(j-(SOM_width/2)))+(eigenvector_2*(i-(SOM_height/2)))))

        
        self.nodes = wt
    
        
        self.trainRecord = [[[-1 for i in range(0)]  for x in range(self.width)] for y in range(self.height)]
        self.colourFlags = [[0 for x in range(self.width)] for y in range(self.height)]
        self.composition_map = np.array(self.trainRecord)
        
    def train(self, iterations = 1000, training_vector = [[]]):
        time_constant = iterations/log(self.radius)
        delta_nodes = scipy.array([[[0.0 for i in range(self.FV_size)] for x in range(self.width)] for y in range(self.height)])
        if self.batch:
            print "batch"
            for i in range(1, iterations+1):
                delta_nodes.fill(0)
                decaying_radius = self.radius * exp(-1.0 * i/time_constant)#radius, learning rate, and update equations from: http://www.ai-junkie.com/ann/som/som1.html and http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3105762/
    
                const = 2 * (decaying_radius**2)
                decaying_learning_rate = self.learning_rate * exp(-1.0*i/time_constant)
                sys.stdout.write("\rTraining Iteration: " + str(i) + "/" + str(iterations)+"\n")
        
                for j in range(len(training_vector)):
                    best = self.find_best_match(training_vector[j,2:])
                    self.trainRecord[best[0]][best[1]].append(training_vector[j,0])
                    for loc in self.find_neighborhood(best, decaying_radius):
                
                        influence = exp( (-1.0 * (loc[2]**2))/const)
                        #diff = (training_vector[j] - self.nodes[loc[0]][loc[1]])
                        #inf_lr_diff = (influence * decaying_learning_rate) * (training_vector[j] - self.nodes[loc[0],loc[1]])
                        #self.nodes[loc[0],loc[1]] += (influence * decaying_learning_rate) * (training_vector[j,1:] - self.nodes[loc[0],loc[1]])                    
                        delta_nodes[loc[0],loc[1]] += (influence * decaying_learning_rate) * (training_vector[j,2:] - self.nodes[loc[0],loc[1]])
                
                self.nodes += delta_nodes
        
        else:
            print "nonbatch"
            for i in range(1, iterations+1):
                delta_nodes.fill(0)
                decaying_radius = self.radius * exp(-1.0 * i/time_constant)#radius, learning rate, and update equations from: http://www.ai-junkie.com/ann/som/som1.html and http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3105762/
    
                const = 2 * (decaying_radius**2)
                decaying_learning_rate = self.learning_rate * exp(-1.0*i/time_constant)
                sys.stdout.write("\rTraining Iteration: " + str(i) + "/" + str(iterations)+"\n")
        
                for j in range(len(training_vector)):
                    best = self.find_best_match(training_vector[j,2:])
                    self.trainRecord[best[0]][best[1]].append(j)
                    for loc in self.find_neighborhood(best, decaying_radius):
                
                        influence = exp( (-1.0 * (loc[2]**2))/const)
                        #diff = (training_vector[j,1:] - self.nodes[loc[0]][loc[1]])
                        #inf_lr_diff = (influence * decaying_learning_rate) * (training_vector[j] - self.nodes[loc[0],loc[1]])
                        self.nodes[loc[0],loc[1]] += (influence * decaying_learning_rate) * (training_vector[j,1:] - self.nodes[loc[0],loc[1]])                    
                        #delta_nodes[loc[0],loc[1]] += (influence * decaying_learning_rate) * (training_vector[j,1:] - self.nodes[loc[0],loc[1]])
                
               # self.nodes += delta_nodes
        for i in range(self.height):
            for j in range(self.width):
                if self.trainRecord[i][j]:
                    self.composition_map[i,j] = int(collections.Counter(self.trainRecord[i][j]).most_common()[0][0]) #stores the most common class from the training record for each node on a composition map
        sys.stdout.write("\n")
        

    def u_marix(self, name):
        from PIL import Image
        import colorsys
        avg_dist = scipy.array([[0.0 for x in range(self.width)]  for y in range(self.height)])
        for i in range(self.height):
            for j in range(self.width):
                n = []
                for g in range(i-1,i+1):
                    for h in range(j-1,j+1):
                        if 0<=g<self.height and 0<=h<self.width:
                            n.append(self.FV_distance(self.nodes[i,j],self.nodes[g,h]))
                #n = [self.FV_distance(self.nodes[i,j],self.nodes[i-1,j-1]), self.FV_distance(self.nodes[i,j],self.nodes[i-1,j]),self.FV_distance(self.nodes[i,j],self.nodes[i-1,j+1]),self.FV_distance(self.nodes[i,j],self.nodes[i,j-1]),self.FV_distance(self.nodes[i,j],self.nodes[i,j+1]), self.FV_distance(self.nodes[i,j],self.nodes[i+1,j-1]), self.FV_distance(self.nodes[i,j],self.nodes[i+1,j]), self.FV_distance(self.nodes[i,j],self.nodes[i+1,j+1])]
                avg_dist[i,j] = np.mean(np.array(n))
        u = avg_dist     
        print u
        
        max = 0
        for i in range(len(u)):
            for j in range(len(u[i])):
                if u[i,j]>max:
                    max = u[i,j]
        print max
        imag = u
        img = Image.new("L", (S.width, S.height))
        for r in range(len(u)):
            for c in range(len(u[i])):           
                img.putpixel((c,r), int(255 - (255.0 * (u[r,c]/max))))
                imag[r,c] = int(255.0 - (255.0 * (u[r,c]/max)))
        print imag
        img = img.resize((S.width*10, S.height*10),Image.NEAREST)
        img.save(str(name+".png"))  


    def kmer_list(self, dna, k):
        result = []
        for x in range(len(dna) + 1-k):
            result.append(dna[x:x+k])
        return result


    
    def FV_distance(self, FV_1, FV_2):
        return (sum((FV_1 - FV_2)**2))**0.5


    
    def find_best_match(self, target_FV):
        distances = (((self.nodes - target_FV)**2).sum(axis=2))**0.5 #produces a matrix corresponding to the distances of each node to the target FV
        i,j = np.unravel_index(distances.argmin(),distances.shape) #returns the indices of the node with the minimum distance in (i,j)
        return (i, j)


    
    def find_neighborhood(self, pt, dist):
        min_y = max(int(pt[0] - dist), 0)
        max_y = min(int(pt[0] + dist), self.height)
        min_x = max(int(pt[1] - dist), 0)
        max_x = min(int(pt[1] + dist), self.width)
        neighbors = []
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                dist = abs(y - pt[0]) + abs(x - pt[1])
                neighbors.append((y,x,dist))
        return neighbors


    
    def classify(self, filename):
        [classData, classheaders] = self.generate_dataVectors(filename)
        classData_np= np.array(classData)
        x = [1,1]
        classes = [[[-1 for i in range(0)]  for x in range(self.width)] for y in range(self.height)]
        for k in range(len(classData_np)):
            BMU = self.find_best_match(classData_np[k,2:])
        
            classes[BMU[0]][BMU[1]].append(classData_np[k,1])
            
       
        return [classes,classheaders]
            
    def produce_classes_map(self, c, name, boolean = True):
        from PIL import Image
        import colorsys
        N = len(c)
        if N==1:
            RGB_tuples = [(1,1,1)]
        else:
            HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
            RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        RGB_tuples_scales = tuple(tuple(int(i*255) for i in inner) for inner in RGB_tuples)

        img2 = Image.new("RGB", (1, len(c)))
        for i in range(len(c)):
            img2.putpixel((0,i),RGB_tuples_scales[i])
        img2 = img2.resize((50, len(c)*50),Image.NEAREST)
        draw = ImageDraw.Draw(img2)
        #font = ImageFont.truetype("sans-serif.ttf", 16)
        font = ImageFont.load_default()
        for i in range(len(c)):
            draw.text((0, (i*50)+5),str(i+1),(255,255,255),font=font)
        img2.save("colour_index_"+name+".png")
        
        img = Image.new("RGB", (S.width, S.height))
      
        maxlens = [0 for x in range(len(c))]
        for t in range(len(c)):
            for y in range(len(c[t])):
                for u in range(len(c[t][y])):
                    if (maxlens[t] < len(c[t][y][u])):
                        maxlens[t] = len(c[t][y][u])
       
        print "Saving Image: "+ name +".png..."
        for j in range(S.height):
            for i in range(S.width):
                if (boolean == False):
                    list_tuples = [(tuple(int(inner*len(c[t][j][i])/maxlens[t]) for inner in RGB_tuples_scales[t])) for t in range(len(c))]
                else:
                    list_tuples = [(tuple(int(inner*(1 if c[t][j][i] else 0)) for inner in RGB_tuples_scales[t])) for t in range(len(c))]
                img.putpixel((i,j), tuple([sum(y) for y in zip(*list_tuples)])) #(int((len(c[t][j][i])/maxlen)*RGB_tuples[t])))
        img = img.resize((S.width*10, S.height*10),Image.NEAREST)
        img.save(name +".png")


    def find_class(self, class_name, header_list):
        index = header_list.index(class_name)
        return index
        
    def colour(self, u, name):
        max = 0
        for i in range(len(u)):
            for j in range(len(u[i])):
                if u[i,j]>max:
                    max = u[i,j]
        img = Image.new("L", (S.width, S.height))
        for r in range(len(u)):
            for c in range(len(u[i])):           
                img.putpixel((c,r), int(255*(u[i,j]/max)))
        img = img.resize((S.width*10, S.height*10),Image.NEAREST)
        img.save(str(name+".png"))            
             
    def colour_map(self):
        from PIL import Image
        for q in range(len(self.trainRecord)):
            for v in range(len(self.trainRecord[q])):
                headers = [self.trainHeaders[i] for i in self.trainRecord[q][v][:]]
                genuses = [headers[y].split()[0] for y in range(len(headers))]
                
                if self.all_identical(genuses):
                    self.colourFlags[q][v] = 1
        #print "Saving Image: colour_map.png..."
        img = Image.new("L", (S.width, S.height))
        for r in range(S.height):
            for c in range(S.width):           
                img.putpixel((c,r), int(255*self.colourFlags[r][c]))
        img = img.resize((S.width*10, S.height*10),Image.NEAREST)
        img.save("colour_map.png")            



    def all_identical(self, items):
        return all(x == items[0] for x in items)


    
    def generate_dataVectors(self, filenames):
        fv = []
    
        for y in range(len(filenames)):
            fname = filenames[y]
            headers = []
            if fname is None:
                fi = sys.stdin
            else:
                fi = open(fname)
            first = fi.readline()
            if first.startswith(">") and (first != 1):
                headers.append(first.rstrip())
            #next(fi)
            kmer_size = self.kmer
            
            header = ""
            dna    = ""
            window = self.windowsize
            if self.windowed:
                i=0
                for line in fi:
                    
                    if (line.startswith(">") and (line != 1) and len(dna) > 10) or ((len(dna) >= window)):
                        
                        #if (len(dna) == window):
                        if line.startswith(">"):
                            i = i+1
                            headers.append(line.rstrip())
        
                        fv.append([y] + [i-1] + self.generate_FV(dna, kmer_size))
                        
                        dna = ""
                    else:
                        dna+=line.rstrip()
                
                fi.close()
            else:
                for line in fi:
                    if (line.startswith(">") and (line != 1)):
                        
                        if len(dna) > 10:
                            headers.append(line.rstrip())
                            fv.append(self.generate_FV(dna, kmer_size))
                        
                        dna = ""
                    else:
                        dna+=line.rstrip()
            
                fi.close()
        return [fv, headers]

    def write_map(self, name):
        f = open(name+str(self.height)+'x'+str(self.width)+'x'+str(self.FV_size), 'w')
        for i in range(self.height):
            for j in range(self.width):
                for t in range(len(self.nodes[i][j])):
                    f.write(str(self.nodes[i][j][t])+" ")
                
                f.write("\n")
            f.write("\n")
        f.close()
        print "Map saved"
        
    def read_map(self, name, height, width, features):
        a = np.loadtxt(name)
        b = np.reshape(a,(height, width, features))
        self.nodes =  b
        print "Map read"
    
    def generate_FV(self, dna, kmer_size): #generates the feature vector for a certain dna and a certain kmer size
        #print len(dna)
        bases=['A','T','G','C']
        kmers=[''.join(p) for p in itertools.product(bases, repeat = kmer_size)]
        my_list = self.kmer_list(dna, kmer_size)
        c = collections.Counter(my_list) #http://pythonforbiologists.com/index.php/kmer-counting-business-card/
        c2 = []
        for i in range(len(kmers)):
            km = c[kmers[i]]
            perc = (km/len(my_list))
            c2.append(perc)
        return c2

    def update_attributes(self):
        self.height = len(self.nodes)
        self.width = len(S.nodes[0])
        self.FV_size = len(S.nodes[0][0])
        self.radius = max(self.height, self.width)/self.radiusfactor
        self.kmer = int(math.log(self.FV_size, 4))
    
    def contigs_classes(self, classes, headers):
        print len(headers)
        contigs =  np.array([[-1 for i in range(0)]  for x in range(len(headers))])
        for k in range(len(headers)):
            for i in range(len(classes)):
                for j in range(len(classes[i])):
                    if k in classes[i][j]:
                        np.append(contigs[k], (i,j))
        return contigs
        
    @staticmethod
    def evaluate_composition(list):
        types = []
        types_counts = []
        for i in range(len(list)):
            if list[i] not in types:
                types.append(list[i])
        for j in range(len(types)):
            types_counts.append((types[j],list.count(types[j])))
        return types_counts    
      
        
    def check_chimerity(self, ContigsClasses):
        contig_composition = np.array([[0 for i in range(len(ContigsClasses[l]))] for l in range(len(ContigsClasses))])
        for i in range(len(ContigsClasses)):
            for j in range(len(ContigsClasses[i])):
                [n,m] = ContigsClasses[i,j]
                contig_composition[i,j] = self.composition_map[n,m] 
        for i in range(len(contig_composition)):
            contig_composition[i] = SOM.evaluate_composition(contig_composition[i])        
               
        return contig_composition
         
if __name__ == "__main__": 
    p_r = "protozoa.2.1.genomic.fna"
    f_r = "fungi.1.1.genomic.fna"
    b_r = "bacteria.421.1.genomic.fna"
    a_r = "archaea.4.1.genomic.fna"
    v_r = "viral.1.1.genomic.fna"

    letter = "p"
    cut = False
    cutsize = 50

    species = 0

    
    
#    p_r_c = "protozoa-"+str(species)+"-species-"+str(cutsize)+"-size.fna"
#    f_r_c = "fungi-"+str(species)+"-species-"+str(cutsize)+"-size.fna"
#    b_r_c = "bacteria-"+str(species)+"-species-"+str(cutsize)+"-size.fna"
#    a_r_c = "archaea-"+str(species)+"-species-"+str(cutsize)+"-size.fna"
#    v_r_c = "viral-"+str(species)+"-species-"+str(cutsize)+"-size.fna"

    #p_r_c = "protozoa-"+str(species)+"-species.fna"
    #f_r_c = "fungi-"+str(species)+"-species.fna"
    #b_r_c = "bacteria-"+str(species)+"-species.fna"
    #a_r_c = "archaea-"+str(species)+"-species.fna"
    #v_r_c = "viral-"+str(species)+"-species.fna"

    p_r_c = "protozoa-cut-"+str(cutsize)+".fna"
    f_r_c = "fungi-cut-"+str(cutsize)+".fna"
    b_r_c = "bacteria-cut-"+str(cutsize)+".fna"
    a_r_c = "archaea-cut-"+str(cutsize)+".fna"
    v_r_c = "viral-cut-"+str(cutsize)+".fna"

    p = "Protozoa.fna"
    f = "total_fungus.fna"
    b = "Bacteria_total.fna"
    a = "Archaea_total.fna"
    v = "Virus_total.fna"
    
#    p1 = "Protozoa.fna"
#    p2 = "protozoa-test.fna"
#    f1 = "fungus.fna"
#    f2 = "fungus2.fna"    
#    f3 = "fungus3.fna"    
#    b1 = "Bacteria.fna"
#    b2 = "Bacteria2.fna"
#    a1 = "Archaea-cg-1.fna"
#    a2 = "Archaea-cg-2.fna"
#    a3 = "Archaea-cg-3.fna"
#    a4 = "Archaea-cg-4.fna"
#    v1 = "Virus1.fna"
#    v2 = "Virus2.fna"    
#    v3 = "Virus3.fna"
#    v4 = "Virus4.fna"
#    v5 = "Virus5.fna"
    
    p1 = "Protozoa-cg-1.fna"
    p2 = "Protozoa-chr-1.fna"
    p3 = "Protozoa-chr-2.fna"
    p4 = "Protozoa-chr-3.fna"
    p5 = "Protozoa-chr-4.fna"
    f1 = "Fungi-cg-1.fna"
    f2 = "Fungi-chr-1.fna"    
    f3 = "Fungi-chr-2.fna"   
    b1 = "Bacteria-cg-1.fna"
    b2 = "Bacteria-cg-2.fna"
    b3 = "Bacteria-cg-3.fna"    
    b4 = "Bacteria-cg-4.fna"
    b5 = "Bacteria-cg-5.fna"

    a1 = "Archaea-cg-1.fna"
    a2 = "Archaea-cg-2.fna"
    a3 = "Archaea-cg-3.fna"
    a4 = "Archaea-cg-4.fna"
    v1 = "Viral-cg-1.fna"
    v2 = "Viral-cg-2.fna"    
    v3 = "Viral-cg-3.fna"
    v4 = "Viral-cg-4.fna"
    v5 = "Viral-cg-5.fna"    
    v6 = "Viral-cg-6.fna"

    p_t = "protozoa-test.fna"
    f_t = "fungus-test.fna"
    b_t = "Bacteria-test.fna"
    a_t = "Archaea-test.fna"
    v_t = "Virus-test-total.fna"

    p_t2 = "protozoa-cut-6.fna"
    f_t2 = "fungi-cut-6.fna"
    b_t2 = "bacteria-cut-6.fna"
    a_t2 = "archaea-cut-6.fna"
    v_t2 = "viral-cut-6.fna"
    
    pfbva = "Protozoa_fungus_bacteria_virus_and_archea.fna"
    cont = "filtered_contigs.fa"
    if letter == "r":
        if cut:
            u = 0
            S = SOM([p_r_c, f_r_c, b_r_c, a_r_c, v_r_c])
        else:
            S = SOM([p_r, f_r, b_r, a_r, v_r])
    if letter == "t":
        S = SOM([p_t, f_t, b_t, a_t, v_t])
        
    if letter == "p":
        #S = SOM(["test.fa"])        
        #S = SOM([f1,f2,b1,b2])
        #S = SOM([p1, p2, p3, p4, p5, f1, f2, f3, b1, b2, b3, b4, b5, a1, a2, a3, a4, v1, v2, v3, v4, v5, v6, v_t])
        #S = SOM([a1,a2,a3,a4])
        S = SOM([p, f, b, a, v])
    if letter == "cont":
        S = SOM([cont])
    #S = SOM([a, v])
    #S = SOM([pfbva])
    #S = SOM([b, f])
    #S = SOM("Bacteria_and_fungus.fna")
    #S = SOM(["test2.fa"])
    #S = SOM("contigs.fa")
    #S.produce_classes_map([classes3, classes4], "kmer"+str(S.kmer)+"-B&A-"+"classesmap-BvsA-files-before-training" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))

    S.train(S.iterations, S.trainV)
    #print S.trainRecord
    
   
    #S.write_map("trained-map-contigs-"+str(S.kmer)+"-window-"+str(S.windowsize))
    S.write_map("trained-map-P&B&F&A&V_all-species-"+letter+"-"+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+str(S.iterations)+"iter"+"-LR"+str(S.learning_rate) + ("-batch" if S.batch else "-nonbatch-") +"radiusfactor-"+str(S.radiusfactor) + ("-windowed-"+str(S.windowsize) if S.windowed else "")+"-")
    #S.write_map("trained-map-P&B&F&A&V_"+letter+"-"+("c"+str(cutsize)+"-" if cut else "-")+str(S.iterations)+"iter"+"-LR"+str(S.learning_rate) + ("-batch" if S.batch else "-nonbatch-") +"radiusfactor-"+str(S.radiusfactor) + ("-windowed-"+str(S.windowsize) if S.windowed else "")+"-")

    #S.read_map("trained-map-P&B&F&A&V_p--560iter-LR0.001-batchradiusfactor-3-windowed-10000-32x53x256",32,53,256)
    #S.update_attributes()
    
    #S.u_marix("u-matrix_cont_"+str(S.windowsize))
      
    #[classes11, headers1] = S.classify([p_r_c])
    #[classes12, headers2] = S.classify([f_r_c])
    #[classes13, headers3] = S.classify([b_r_c])
    #[classes14, headers4] = S.classify([a_r_c])
    #[classes15, headers5] = S.classify([v_r_c])  
        
    #[classes21, headers1] = S.classify([p_t2])
    #[classes22, headers2] = S.classify([f_t2])
    #[classes23, headers3] = S.classify([b_t2])
    #[classes24, headers4] = S.classify([a_t2])
    #[classes25, headers5] = S.classify([v_t2])
    
#    [classes31, headers1] = S.classify([p_t])
#    [classes32, headers2] = S.classify([f_t])
#    [classes33, headers3] = S.classify([b_t])
#    [classes34, headers4] = S.classify([a_t])
#    [classes35, headers5] = S.classify([v_t])
          
#    [classes41, headers1] = S.classify([p_r])
#    [classes42, headers2] = S.classify([f_r])
#    [classes43, headers3] = S.classify([b_r])
#    [classes44, headers4] = S.classify([a_r])
#    [classes45, headers5] = S.classify([v_r])
   
     
#    [classes51, headers1] = S.classify([p])
#    [classes52, headers2] = S.classify([f])
#    [classes53, headers3] = S.classify([b])
#    [classes54, headers4] = S.classify([a])
#    [classes55, headers5] = S.classify([v])
    
   # [classes60, headers60] = S.classify([p1, p2, p3, p4, p5])
#    [classes61, headers1] = S.classify([p2])
#    [classes62, headers1] = S.classify([p3])
#    [classes63, headers1] = S.classify([p4])
#    [classes64, headers1] = S.classify([p5])
    
  #  [classes65, headers2] = S.classify([f1, f2, f3])
#    [classes66, headers2] = S.classify([f2])
#    [classes67, headers2] = S.classify([f3])
 #   [classes68, headers3] = S.classify([b1, b2, b3, b4, b5])
#    [classes69, headers3] = S.classify([b2])
#    [classes70, headers3] = S.classify([b3])
#    [classes71, headers3] = S.classify([b4])
#    [classes72, headers3] = S.classify([b5])
   # [classes73, headers4] = S.classify([a1, a2, a3, a4])
    #[classes73a, headers73a] = S.classify([a1])
    #[classes74, headers4] = S.classify([a2])
    #[classes75, headers4] = S.classify([a3])
    #[classes76, headers4] = S.classify([a4])
 #   [classes77, headers5] = S.classify([v1, v2, v3, v4, v5, v_t])
#    [classes78, headers5] = S.classify([v2]) 
#    [classes79, headers5] = S.classify([v3])
#    [classes80, headers5] = S.classify([v4])
#    [classes81, headers5] = S.classify([v5]) 
#    [classes82, headers5] = S.classify([v6])
#    [classes83, headers5] = S.classify([v_t])  
#    [classes100, headers100] = S.classify(["filtered_contigs.fa"])
    [classes60, headers60] = S.classify([p, f, b, a, v])
    S.produce_classes_map([classes60], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_p"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    der = S.check_chimerity(S.contigs_classes(classes60, headers60))
    for i in range(len(headers60)):
        print headers60[i] +" : " + str(len(der[i]))+"\n"
        
    sys.exit()
    #S.produce_classes_map([classes41, classes42, classes43, classes44, classes45], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes21, classes22, classes23, classes24, classes25], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_t2"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes31, classes32, classes33, classes34, classes35], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_t"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes41, classes42, classes43, classes44, classes45], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_r"+str(species)+"-" if cut else "-")+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes51, classes52, classes53, classes54, classes55], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_p"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes62, classes63, classes65, classes66 ], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-F&B_species-"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_F&B"+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes651, classes64], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-F&B(test)_species-"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_F&B"+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))

    #S.produce_classes_map([classes64, classes70 ], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-F&A_species-"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
       
    S.produce_classes_map([classes60, classes65, classes68, classes73, classes77], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
       
       
    #S.produce_classes_map([classes11, classes12, classes13, classes14, classes15], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    S.produce_classes_map([classes100], "kmer" + str(S.kmer) + "_contigs-20mb-against-P-B-F-A-V_species-all-"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-map")
    
    #S.produce_classes_map([classes67, classes68, classes69,classes70 ], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_pa"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_pa"+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))

    #[classes6, headers6] = S.classify("Uniprot-Amylase-contigs.extract.fa")
    #[classes7, headers7] = S.classify("Uniprot-Cellulase-contigs.extract.fa")
    #[classes8, headers8] = S.classify("Uniprot-Laminarinase-contigs.extract.fa")
    #[classes9, headers9] = S.classify("Uniprot-Alginate_Lyase-contigs.extract.fa")
    
    #[classes, headers] = S.classify("test.fa")
    #[classes2, headers2] = S.classify("test.fa")
    #[classes3, headers3] = S.classify("test.fa")


    #S.produce_classes_map([classes1], "kmer"+str(S.kmer)+"_protozoa-against-P-B-F-A-V-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes2], "kmer"+str(S.kmer)+"_fungi-against-P-B-F-A-V-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes3], "kmer"+str(S.kmer)+"_bacteria-against-P-B-F-A-V-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes4], "kmer"+str(S.kmer)+"_archaea-against-P-B-F-A-V-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes5], "kmer"+str(S.kmer)+"_virus-against-P-B-F-A-V-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))

    #S.produce_classes_map([classes6], "kmer"+str(S.kmer)+"_Magda-Amylase-against-P-B-F-A-V-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes7], "kmer"+str(S.kmer)+"_Magda-Cellulase-against-P-B-F-A-V-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes8], "kmer"+str(S.kmer)+"_Magda-Laminarinase-against-P-B-F-A-V-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes9], "kmer"+str(S.kmer)+"_Magda-Alginate-against-P-B-F-A-V-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))

    
    #S.produce_classes_map([classes1, classes2, classes3, classes4, classes5], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V-"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV-files" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes11, classes12, classes13, classes14, classes15], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_r-"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_r-files" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))


#    S.produce_classes_map([classes21], "kmer"+str(S.kmer)+"_protozoa-against-P-B-F-A-V-t2-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)
#    S.produce_classes_map([classes22], "kmer"+str(S.kmer)+"_fungi-against-P-B-F-A-V-t2-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)
#    S.produce_classes_map([classes23], "kmer"+str(S.kmer)+"_bacteria-against-P-B-F-A-V-t2-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)
#    S.produce_classes_map([classes24], "kmer"+str(S.kmer)+"_archaea-against-P-B-F-A-V-t2-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)
#    S.produce_classes_map([classes25], "kmer"+str(S.kmer)+"_virus-against-P-B-F-A-V-t2-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)

    #S.produce_classes_map([classes31], "kmer"+str(S.kmer)+"_protozoa-against-P-B-F-A-V-t-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)
    #S.produce_classes_map([classes32], "kmer"+str(S.kmer)+"_fungi-against-P-B-F-A-V-t-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)
    #S.produce_classes_map([classes33], "kmer"+str(S.kmer)+"_bacteria-against-P-B-F-A-V-t-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)
    #S.produce_classes_map([classes34], "kmer"+str(S.kmer)+"_archaea-against-P-B-F-A-V-t-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)
    #S.produce_classes_map([classes35], "kmer"+str(S.kmer)+"_virus-against-P-B-F-A-V-t-single-class-map" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""), False)

    #S.produce_classes_map([classes21, classes22, classes23, classes24, classes25], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_t2-"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("-c"+str(cutsize) if cut else "")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes31, classes32, classes33, classes34, classes35], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_t-"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("-c"+str(cutsize) if cut else "")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))

    #S.produce_classes_map([classes10], "kmer"+str(S.kmer) + "_contigs-20mb-against-P-B-F-A-V_t"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-map", False)
    #S.produce_classes_map([classes11, classes12, classes13, classes14, classes15], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_"+letter+("c"+str(cutsize)+"-" if cut else "-")+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("-c"+str(cutsize) if cut else "")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))

    
    #S.produce_classes_map([classes2, classes3], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-B&F(test)-"+str(S.iterations)+"iter-kmerperc-" + ("batch" if S.batch else "nonbatch") + "-classesmap-BvsF-files" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    #S.produce_classes_map([classes4, classes5], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-A&V-"+str(S.iterations)+"iter-kmerperc-" + ("batch" if S.batch else "nonbatch") + "-classesmap-AvsV-files" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
        
    #S.produce_classes_map([classes12, classes15], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-F_r&V_r-"+str(S.iterations)+"iter-kmerperc-" + ("batch" if S.batch else "nonbatch") + "-classesmap-F_rvsV_r-files" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))

    #S.produce_classes_map([classes6, classes7, classes8, classes9], "kmer"+str(S.kmer)+"-Magda-"+str(S.iterations)+"iter-kmerperc-" + ("batch" if S.batch else "nonbatch") + "-classesmap-PvsBvsFvsAvsV-files" +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    
    #S.produce_classes_map([classes2, classes3], "Classes_B&F-map")
    #S.produce_classes_map([classes4], "Classes_F2-against-B&F-map")    
    
    #colour_single_class("Penicillium marneffei", classes_map, headers_list, col)
    #S.colour_map()


    #from PIL import Image
    #print "Saving Image: som_test_colors.png..."
    #img = Image.new("RGB", (S.width, S.height))
    #for r in range(S.height):
    #    for c in range(S.width):
    #        img.putpixel((c,r), (int(S.nodes[r][c][0]), int(S.nodes[r][c][1]), int(S.nodes[r][c][2])))
    #img = img.resize((S.width*10, S.height*10),Image.NEAREST)
    #img.save("som_test_colors.png")
