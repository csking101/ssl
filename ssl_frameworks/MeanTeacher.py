import torch


class MeanTeacher:
    def __init__(self, architecture, alpha, optimizer, initial_adaptive_weighing_coefficient,supervised_loss,unsupervised_loss,device='cpu',*args):
        
        self.alpha = alpha
        self.adaptive_weighing_coefficient = initial_adaptive_weighing_coefficient #This is the beta term in ramp up
        
        
        #In Architecture you can pass the class constructor of the model that you need
        print("Initializing student model")
        self.student_model = architecture(*args).to(device)
        print("Initializing teacher model")
        self.teacher_model = architecture(*args).to(device)
        
        self.optimizer = optimizer(self.student_model.parameters(),lr=0.001)#This might throw an error, but you can replace with the module list of encoders and decoders
        
        self.supervised_loss = supervised_loss
        self.unsupervised_loss = unsupervised_loss
        
    def _update_ramp_up_coefficient(self):
        pass
        
    def _update_teacher_weights(self):
        for student_weight, teacher_weight in zip(self.student_model.parameters(), self.teacher_model.parameters()):
            with torch.no_grad(): #Save memory with not creating the computational graph
                teacher_weight = self.alpha * teacher_weight + (1 - self.alpha) * student_weight
    
    def train(self,x,ground_truth=None): #This is for individual samples, not sure if it will work with batch or not
        self.student_model.train()
        self.teacher_model.eval()

        self.optimizer.zero_grad()

        student_prediction = self.student_model(x)
        teacher_prediction = self.teacher_model(x)
        
        loss = self.unsupervised_loss(student_prediction, teacher_prediction)
        
        if ground_truth is not None:
            loss = loss + self.adaptive_weighing_coefficient * self.supervised_loss(student_prediction, ground_truth)
            
#         print(f"Current loss is {loss.item()}")
            
        loss.backward()#See if this is correct
        self.optimizer.step()
        
        self._update_teacher_weights()
        self._update_ramp_up_coefficient()
        
        return loss
    
    def test_models(self,x,ground_truth):
        self.student_model.eval()
        self.teacher_model.eval()
        
        with torch.no_grad():
            student_prediction = self.student_model(x)
            teacher_prediction = self.teacher_model(x)
            
            c_loss = self.unsupervised_loss(student_prediction, teacher_prediction)
            ce_loss = self.supervised_loss(student_prediction, ground_truth)
            
            return (c_loss, ce_loss)
            
