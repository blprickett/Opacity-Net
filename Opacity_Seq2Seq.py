import numpy as np
import scipy.stats as sp
import keras
import seq2seq
from seq2seq.models import Seq2Seq
from random import shuffle
from keras.optimizers import RMSprop, Adam
from sys import argv

##USER PARAMETERS##				
REPS = int(argv[1]) #20 
EPOCHS = int(argv[2]) #10
FEATURES = argv[3] #"byHand" or "oneHot"

##TRAINING DATA##
#Dictionary, mapping segments to their feature bundles
if FEATURES == "byHand":
	FEAT_CONVERT = {		 #syllabic, coronal, 	anterior, 	dorsal, high
						"t": [-1, 		1,			1,			0,		0],
						"T": [-1, 		1,			-1,			0,		0],
						"k": [-1, 		0,			0,			1,		0],
						"i": [1, 		0,			0,			0,		1],
						"a": [1, 		0,			0,			0,		-1],
						"E": [0,		0,			0,			0,		0], #(Empty symbol)
					}
	FEAT_NUM = len(list(FEAT_CONVERT.values())[0])
elif FEATURES == "oneHot":
	FEAT_CONVERT = {		 
						"t": [1, 		0,			0,			0,		0],
						"T": [0, 		1,			0,			0,		0],
						"k": [0, 		0,			1,			0,		0],
						"i": [0, 		0,			0,			1,		0],
						"a": [0, 		0,			0,			0,		1],
						"E": [0,		0,			0,			0,		0], #(Empty symbol)
					}
	FEAT_NUM = len(list(FEAT_CONVERT.values())[0])
else:
	exit("Error in feature type! (choose 'byHand' or 'oneHot')")

#Frequencies for each trial type (can manipulate to compare to Jarosz 2016 results)
interact_Quant = 18
pal_Quant = 18
hait_Quant = 18
faith_Quant = 18

#Construct the data:
#Bleeding
bleeding_TD = []								#Quant	UR	->	SR
bleeding_TD +=	[["tiE", "TiE"]] * pal_Quant	#18		ti 	->	Ti
bleeding_TD +=	[["taE", "taE"]] * faith_Quant	#18		ta	->	ta		
bleeding_TD +=	[["tia", "taE"]] * interact_Quant#18	tia	->	ta
bleeding_TD +=	[["kiE", "kiE"]] * faith_Quant	#18		ki	->	ki
bleeding_TD +=	[["kaE", "kaE"]] * faith_Quant	#18		ka	-> 	ka
bleeding_TD +=	[["kia", "kaE"]] * hait_Quant	#18		kia	->	ka

#Feeding
feeding_TD = []									#Quant	UR	->	SR
feeding_TD +=	[["tiE", "TiE"]] * pal_Quant	#18		ti 	->	Ti
feeding_TD +=	[["taE", "taE"]] * faith_Quant	#18		ta	->	ta		
feeding_TD +=	[["tai", "TiE"]] * interact_Quant#18	tai	->	Ti
feeding_TD +=	[["kiE", "kiE"]] * faith_Quant	#18		ki	->	ki
feeding_TD +=	[["kaE", "kaE"]] * faith_Quant	#18		ka	-> 	ka
feeding_TD +=	[["kai", "kiE"]] * hait_Quant	#18		kai	->	ki

#Counterbleeding
counter_bleeding_TD = []								#Quant	UR	->	SR
counter_bleeding_TD +=	[["tiE", "TiE"]] * pal_Quant	#18		ti 	->	Ti
counter_bleeding_TD +=	[["taE", "taE"]] * faith_Quant	#18		ta	->	ta		
counter_bleeding_TD +=	[["tia", "TaE"]] * interact_Quant#18	tia	->	ta
counter_bleeding_TD +=	[["kiE", "kiE"]] * faith_Quant	#18		ki	->	ki
counter_bleeding_TD +=	[["kaE", "kaE"]] * faith_Quant	#18		ka	-> 	ka
counter_bleeding_TD +=	[["kia", "kaE"]] * hait_Quant	#18		kia	->	ka

#Counterfeeding
counter_feeding_TD = []									#Quant	UR	->	SR
counter_feeding_TD +=	[["tiE", "TiE"]] * pal_Quant	#18		ti 	->	Ti
counter_feeding_TD +=	[["taE", "taE"]] * faith_Quant	#18		ta	->	ta		
counter_feeding_TD +=	[["tai", "tiE"]] * interact_Quant#18	tai	->	Ti
counter_feeding_TD +=	[["kiE", "kiE"]] * faith_Quant	#18		ki	->	ki
counter_feeding_TD +=	[["kaE", "kaE"]] * faith_Quant	#18		ka	-> 	ka
counter_feeding_TD +=	[["kai", "kiE"]] * hait_Quant	#18		kai	->	ki

#Dictionary mapping pattern labels to sets of training data:
PATTERNS = {"Bleeding":bleeding_TD, "Feeding":feeding_TD, "Counterbleeding":counter_bleeding_TD,
				"Counterfeeding":counter_feeding_TD}

				
##TEST DATA##
#(forced choice, like the experiment -> [UR, correct SR, wrong SR]):				
raw_test_data = {"Faithful":{	"Bleeding": [["taE", "taE", "TaE"], ["kiE", "kiE", "TiE"], ["kaE", "kaE", "TaE"]],
								"Feeding": [["taE", "taE", "TaE"], ["kiE", "kiE", "TiE"], ["kaE", "kaE", "TaE"]],
								"Counterbleeding": [["taE", "taE", "TaE"], ["kiE", "kiE", "TiE"], ["kaE", "kaE", "TaE"]],
								"Counterfeeding": [["taE", "taE", "TaE"], ["kiE", "kiE", "TiE"], ["kaE", "kaE", "TaE"]]
							  },
				"Hiatus":{	"Bleeding": [["kia", "kaE", "kia"]],
							"Feeding": [["kai", "kiE", "kai"]],
							"Counterbleeding": [["kia", "kaE", "kia"]],
							"Counterfeeding": [["kai", "kiE", "kai"]]
						  },
				"Palatalizing":{	"Bleeding": [["tiE", "TiE", "tiE"]],
								"Feeding": [["tiE", "TiE", "tiE"]],
								"Counterbleeding": [["tiE", "TiE", "tiE"]],
								"Counterfeeding": [["tiE", "TiE", "tiE"]]
							  },
				"Interacting":{	"Bleeding": [["tia", "taE", "TaE"]],
								"Feeding": [["tai", "TiE", "tiE"]],
								"Counterbleeding": [["tia", "TaE", "taE"]],
								"Counterfeeding": [["tai", "tiE", "TiE"]]
							  }
			}

#Organize the test data:
UR_test_data = {}
correct_SR_test_data = {}
wrong_SR_test_data = {}
for trial_type in raw_test_data.keys():
	UR_test_data[trial_type] = {}
	correct_SR_test_data[trial_type] = {}
	wrong_SR_test_data[trial_type] = {}
	for patt in raw_test_data[trial_type].keys():
		URs = [datum[0] for datum in raw_test_data[trial_type][patt]]
		corr_SRs = [datum[1] for datum in raw_test_data[trial_type][patt]]
		wrng_SRs = [datum[2] for datum in raw_test_data[trial_type][patt]]
		UR_test_data[trial_type][patt] = [[FEAT_CONVERT[seg] for seg in ur] for ur in URs] 
		correct_SR_test_data[trial_type][patt] = [[FEAT_CONVERT[seg] for seg in sr]	for sr in corr_SRs]
		wrong_SR_test_data[trial_type][patt] = [[FEAT_CONVERT[seg] for seg in sr]	for sr in wrng_SRs]		

##SIMULATIONS##
curve_file = open("Curve_File (NN-ForcedChoice).csv", "w")
curve_by_trialType = {tt:{p:[] for p in raw_test_data[tt].keys()} for tt in raw_test_data.keys()}		
for patt in PATTERNS.keys():
	print("Compiling Training Data for " + patt)
	
	#Separate input from output and convert to feature activations:
	raw_X = [datum[0] for datum in PATTERNS[patt]]
	raw_Y = [datum[1] for datum in PATTERNS[patt]]
	ordered_X = [[FEAT_CONVERT[seg] for seg in word] for word in raw_X]
	ordered_Y = [[FEAT_CONVERT[seg] for seg in word] for word in raw_Y]
	
	#Go through reps:
	accuracies = {trial_type:[] for trial_type in UR_test_data.keys()}
	for rep in range(REPS):
		print ("Rep: " + str(rep)+" "+patt)
	
		#Shuffle training data:
		indexes = list(range(len(ordered_X)))
		shuffle(indexes)
		X = np.array([ordered_X[i] for i in indexes])
		Y = np.array([ordered_Y[i] for i in indexes])
		
		#Create the model object:
		model = Seq2Seq(input_dim=FEAT_NUM, hidden_dim=FEAT_NUM*3, output_length=3, output_dim=FEAT_NUM, depth=2)
		my_optimizer = RMSprop(lr=0.005)
		model.compile(	loss="mse", 
				optimizer=my_optimizer,
				metrics=['accuracy']
			)
			
		this_curve = []
		for ep in range(EPOCHS):
		#Train the model on epoch at a time, 
		#so we can give it a forced-choice task at each step:
			print ("Epoch: "+str(ep))
			hist = model.fit(
						X, Y,
						epochs=1,
						batch_size=X.shape[0]
				 )
			this_curve.append(hist.history["acc"][-1])
			for trial_type in accuracies.keys():
				corr_loss = model.evaluate(		x=np.array(UR_test_data[trial_type][patt]),
												y=np.array(correct_SR_test_data[trial_type][patt])
											)
				wrong_loss = model.evaluate(	x=np.array(UR_test_data[trial_type][patt]),
												y=np.array(wrong_SR_test_data[trial_type][patt])
											)
				
				#Accuracy, based on Luce choice axiom
				try:
					curve_by_trialType[trial_type][patt][rep].append(wrong_loss[0]/(corr_loss[0]+wrong_loss[0]))
				except:
					curve_by_trialType[trial_type][patt].append([])
					curve_by_trialType[trial_type][patt][rep].append(wrong_loss[0]/(corr_loss[0]+wrong_loss[0]))
				 
		#Record the learning curve:		
		curve_file.write(patt+",")
		curve_file.write(",".join([str(epLoss) for epLoss in this_curve]))
		curve_file.write("\n")
		
		#Evaluate the model on the different kinds of trials (forced choice):
		for trial_type in accuracies.keys():
			if trial_type=="Faithful":
				accs = []
				for ur_index, faithful_ur in enumerate(UR_test_data[trial_type][patt]):
					corr_loss = model.evaluate(		x=np.array([faithful_ur]),
													y=np.array([correct_SR_test_data[trial_type][patt][ur_index]])
												)
					wrong_loss = model.evaluate(	x=np.array([faithful_ur]),
													y=np.array([wrong_SR_test_data[trial_type][patt][ur_index]])
												)
												
					#Accuracy, based on Luce choice axiom
					accs.append(wrong_loss[0]/(corr_loss[0]+wrong_loss[0]))
				acc = np.mean(accs)
			else:
				corr_loss = model.evaluate(		x=np.array(UR_test_data[trial_type][patt]),
												y=np.array(correct_SR_test_data[trial_type][patt])
											)
				wrong_loss = model.evaluate(	x=np.array(UR_test_data[trial_type][patt]),
												y=np.array(wrong_SR_test_data[trial_type][patt])
											)
				
				#Accuracy, based on Luce choice axiom
				acc = wrong_loss[0]/(corr_loss[0]+wrong_loss[0])
			
			#Store accuracies
			accuracies[trial_type].append(acc)
		
		#Delete the model we just trained/tested
		keras.backend.clear_session()
	
	#Save accuracies:
	output_array = [[] for trial_type in accuracies.keys()]
	this_index = 0
	for trial_type in sorted(accuracies.keys()):
		output_array[this_index] = accuracies[trial_type]
		this_index += 1
	print (sorted(accuracies.keys()))
	np.savetxt("Accuracy by Trial Type for "+patt+"+ForcedChoice.csv", np.array(output_array), delimiter=",", newline="\n")

#Save forced-choice learning curves to a file that's easy to read in R:
tt_curve_file = open("Curves by trial type (NN-ForcedChoice).csv", "w")
tt_curve_file.write("Trial Type,Pattern,Rep,Epoch,Accuracy\n")
for trial_type in curve_by_trialType.keys():
	for pattern in curve_by_trialType[trial_type].keys():
		for rep, curve in enumerate(curve_by_trialType[trial_type][pattern]):
			for ep, acc in enumerate(curve):
				tt_curve_file.write(trial_type)
				tt_curve_file.write(",")
				tt_curve_file.write(pattern)
				tt_curve_file.write(",")
				tt_curve_file.write(str(rep))
				tt_curve_file.write(",")
				tt_curve_file.write(str(ep))
				tt_curve_file.write(",")
				tt_curve_file.write(str(acc))
				tt_curve_file.write("\n")
	
##CLOSE FILES/EXIT PROGRAM##
curve_file.close()
tt_curve_file.close()
print ("All done!")