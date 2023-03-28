let view = document.getElementById("view")
let disPlay = document.getElementById("disPlay")
let btnSubmit = document.getElementById("submit")
let list1 = document.getElementById("list1")
let list2 = document.getElementById("list2")
let list3 = document.getElementById("list3")
let list4 = document.getElementById("list4")
let list5 = document.getElementById("list5")

let lists = [list1, list2, list3, list4, list5]
let result = []
let element = []



let Symptoms = ["abdominal_pain", "abnormal_menstruation", "acidity", "acute_liver_failure", "altered_sensorium",
    "anxiety", "back_pain", "belly_pain", "blackheads", "bladder_discomfort", "blister", "blood_in_sputum", "bloody_stool",
    "blurred_and_distorted_vision", "breathlessness", "brittle_nails", "bruising", "burning_micturition", "chest_pain",
    "chills", "cold_hands_and_feet", "coma", "congestion", "constipation", "continuous_feel_of_urine", "continuous_sneezing",
    'cough', "cramps", 'dark_urine', "dehydration", "depression", "diarrhoea", "dyschromic_patches", "distention_of_abdomen",
    'dizziness', "drying_and_tingling_lips", "enlarged_thyroid", "excessive_hunger", 'extra_marital_contacts', 'family_history',
    'fast_heart_rate', 'fatigue', 'fluid_overload', 'fluid_overload_1', 'foul_smell_of_urine', 'headache', 'high_fever', 'hip_joint_pain',
    'history_of_alcohol_consumption', 'increased_appetite', 'indigestion', 'inflammatory_nails', 'internal_itching', 'irregular_sugar_level',
    'irritability', 'irritation_in_anus', 'itching', 'joint_pain', 'knee_pain', 'lack_of_concentration', 'lethargy', 'loss_of_appetite', 'loss_of_balance',
    'loss_of_smell', 'loss_of_taste', 'malaise', 'mild_fever', 'mood_swings', 'movement_stiffness', 'mucoid_sputum', 'muscle_pain', 'muscle_wasting', 'muscle_weakness',
    'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity', 'pain_behind_the_eyes', 'pain_during_bowel_movements', 'pain_in_anal_region', 'painful_walking', 'palpitations',
    'passage_of_gases', 'patches_in_throat', 'phlegm', 'polyuria', 'prominent_veins_on_calf', 'puffy_face_and_eyes', 'pus_filled_pimples', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'red_sore_around_nose', 'red_spots_over_body', 'redness_of_eyes', 'restlessness', 'runny_nose', 'rusty_sputum', 'scurrying', 'shivering',
    'silver_like_dusting', 'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech', 'small_dents_in_nails', 'spinning_movements', 'spotting_urination', 'stiff_neck',
    'stomach_bleeding', 'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints', 'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremities',
    'swollen_legs', 'throat_irritation', 'tiredness', " toxic_look_(typhus)", 'ulcers_on_tongue', 'unsteadiness', 'visual_disturbances', 'vomiting', 'watering_from_eyes', 'weakness_in_limbs',
    'weakness_of_one_body_side', 'weight_gain', 'weight_loss', 'yellow_crust_ooze', 'yellow_urine', 'yellowing_of_eyes', 'yellowish_skin', 'prognosis']





function display() {
    let cartona = ""
    for (let i = 0; i < Symptoms.length; i++) {
        cartona += `
            <div class="col-md-4  border-dark rounded-3">
        <h1 name="${Symptoms[i]}" class="fs-4">${[i]}) ${Symptoms[i]}   </h1>
        </div>
            `
        disPlay.innerHTML = (cartona)
    }
}


btnSubmit.addEventListener('click',  function () {
firstSubmit()
 function firstSubmit() {
    for (let i = 0; i < lists.length; i++) {
        element = lists[i];
        for (let i = 0; i < Symptoms.length; i++) {
            if (Symptoms[i] == element.options[element.selectedIndex].value) {
                result.push(i)

            }
        }
    }
    result.length = 5
    console.log(result)
    myfunction(result)
    return result

}
 function  myfunction(result) {
        const dict_values =  result //Pass the javascript variables to a dictionary.
        const s = JSON.stringify(dict_values); // Stringify converts a JavaScript object or value to a JSON string
        console.log(s); // Prints the variables to console window, which are in the JSON format
 $.ajax({
            url:"/test",
            type:"POST",
            contentType:"application/json",
            data: JSON.stringify(s),
            dataType:'json',
  success: function(result) {
    console.log("Result:");
    console.log(result);
  }
  ,
    error: function(data){
        alert("fail");
         console.log(data)
    }
            })
            .done(function(data){
            console.log(data[1])
            $('#output').text(data[1]).show();
            });
            }
})


