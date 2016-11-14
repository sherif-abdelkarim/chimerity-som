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
        self.kmer = 3
        self.windowed = True
        self.windowsize = 50000
        self.boolean = True
        self.iterations = 200
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
        self.composition_map = [[-1 for x in range(self.width)] for y in range(self.height)]
        
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
                    print "entered the if condition...."
                    print int(collections.Counter(self.trainRecord[i][j]).most_common()[0][0])
                    self.composition_map[i][j] = int(collections.Counter(self.trainRecord[i][j]).most_common()[0][0]) #stores the most common class from the training record for each node on a composition map        

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
        
        max = 0
        for i in range(len(u)):
            for j in range(len(u[i])):
                if u[i,j]>max:
                    max = u[i,j]
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
    
    def contigs_classes(self, classes, headers):#creates a list containing the where each contig was classed on the map, where in the index of the contig a list of locations is created
        print "length of headers list is: " + str(len(headers))
        contigs = [[-1 for i in range(0)] for x in range(len(headers))]
        for k in range(len(headers)):
            for i in range(len(classes)):
                for j in range(len(classes[i])):
                    if k in classes[i][j]:
                        contigs[k].append((i,j))
        print "contigs_classes: "
        print str((contigs))
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
    
    def check_chimerity(self, ContigsClasses): #contig_composition is a list of lists, containing for index 1 info about where contig 1 was classified on the map but not the locations of the nodes on the map but which contig was most commonly classified to this node during training.
        contig_composition = [[0 for i in range(len(ContigsClasses[l]))] for l in range(len(ContigsClasses))]
        for i in range(len(ContigsClasses)):
            for j in range(len(ContigsClasses[i])):
                if  ContigsClasses[i][j]:
                    index1, index2 = ContigsClasses[i][j]
                    contig_composition[i][j] = self.composition_map[index1][index1] 
        print "composition_map: "
        print str((self.composition_map))
        return contig_composition
        #for i in range(len(contig_composition)):
            #contig_composition[i] = SOM.evaluate_composition(contig_composition[i])
         
if __name__ == "__main__": 
    p_r = "fasta/protozoa.2.1.genomic.fna"
    f_r = "fasta/fungi.1.1.genomic.fna"
    b_r = "fasta/bacteria.421.1.genomic.fna"
    a_r = "fasta/archaea.4.1.genomic.fna"
    v_r = "fasta/viral.1.1.genomic.fna"

    p = "fasta/Protozoa_total.fna"
    f = "fasta/total_fungus.fna"
    b = "fasta/Bacteria_total.fna"
    a = "fasta/Archaea_total.fna"
    v = "fasta/Virus_total.fna"
    
    pfbva = "Protozoa_fungus_bacteria_virus_and_archea.fna"
    cont = "filtered_contigs.fa"
    
    S = SOM([p, f, b, a, v])
    S.train(S.iterations, S.trainV)
    #print S.trainRecord
    
    #S.write_map("trained-map-contigs-"+str(S.kmer)+"-window-"+str(S.windowsize))
    S.write_map("trained_maps/trained-map-P&B&F&A&V_all-species-"+str(S.iterations)+"iter"+"-LR"+str(S.learning_rate) + ("-batch" if S.batch else "-nonbatch-") +"radiusfactor-"+str(S.radiusfactor) + ("-windowed-"+str(S.windowsize) if S.windowed else "")+"-")
    #S.write_map("trained-map-P&B&F&A&V_"+letter+"-"+("c"+str(cutsize)+"-" if cut else "-")+str(S.iterations)+"iter"+"-LR"+str(S.learning_rate) + ("-batch" if S.batch else "-nonbatch-") +"radiusfactor-"+str(S.radiusfactor) + ("-windowed-"+str(S.windowsize) if S.windowed else "")+"-")

    #S.read_map("trained-map-P&B&F&A&V_p--560iter-LR0.001-batchradiusfactor-3-windowed-10000-32x53x256",32,53,256)
    #S.update_attributes()
    
    #S.u_marix("u-matrix_cont_"+str(S.windowsize))
      
    [classes60, headers60] = S.classify([p, f, b, a, v])
    [classes61, headers61] = S.classify([p])
    [classes62, headers62] = S.classify([f])
    [classes63, headers63] = S.classify([b])
    [classes64, headers64] = S.classify([a])
    [classes65, headers65] = S.classify([v])

    S.produce_classes_map([classes60], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_p"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV"+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    S.produce_classes_map([classes61, classes62, classes63, classes64, classes65], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V"+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV"+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
    contig_composition = S.check_chimerity(S.contigs_classes(classes60, headers60))
    print "contig_composition: "
    print str((contig_composition))
    for i in range(len(headers60)):
        print str(headers60[i]) + ": " + str(contig_composition[i])
    #print "training record: " + str(S.trainRecord)
    sys.exit()
       
    S.produce_classes_map([classes60, classes65, classes68, classes73, classes77], "kmer"+str(S.kmer)+"-LR"+str(S.learning_rate)+"-P&B&F&A&V_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+str(S.iterations)+"iter-kmerperc-batch-classesmap-PvsBvsFvsAvsV_"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-files-radiusfactor-"+str(S.radiusfactor) +("-windowed-"+str(S.windowsize) if S.windowed else "")+("-boolean" if S.boolean else ""))
       
    S.produce_classes_map([classes100], "kmer" + str(S.kmer) + "_contigs-20mb-against-P-B-F-A-V_species-all-"+letter+("c"+str(cutsize)+"-sp-"+str(species)+"-" if cut else "-")+"-map")
    
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
