def train_nlu():
    from rasa_nlu.training_data import load_data
    from rasa_nlu import config
    from rasa_nlu.model import Trainer
    
    print('1st')
    training_data = load_data('D:/chatbot/data/nlu.md')
    print('2nd')
    trainer = Trainer(config.load("D:/chatbot/config.yml"))
    print('3rd')
    trainer.train(training_data)
    print('4th')
    model_directory =     	trainer.persist('D:/chatbot/models/nlu/',
                                      	fixed_model_name="current")

    return model_directory



train_nlu()