(ns predict
  (:require [clojure.edn :as edn]
            [clojure.set :as set]
            [libpython-clj2.python :as py]
            [libpython-clj2.require :refer [require-python]]
            [scicloj.ml.dataset :as ds]
            [tech.v3.dataset.tensor :as dst]))

(require-python '[builtins])
(require-python '[pickle])


(println "Establishing the to-be-predicted dataset")

(def to-be-predicted-ds 
  (let [col-names    [:sepal_length :sepal_width :petal_length :petal_width :species-truth]
        rows         [[5.1 3.5 1.4 0.2 "setosa"]
                      [7.0 3.2 4.7 1.4 "versicolor"]
                      [6.3 3.3 6.0 2.5 "virginica"]]
        seq-of-pairs (->> col-names
                          (map-indexed vector)
                          (map (fn [[idx col-name]]
                                 [col-name (map #(nth % idx) rows)])))]
    (ds/dataset seq-of-pairs)))
(println "to-be-predicted-ds" (ds/shape to-be-predicted-ds))
(println (ds/head to-be-predicted-ds))


(println "Featurising the dataset")

(def numeric-ds (-> to-be-predicted-ds
                    (ds/drop-columns :species-truth)
                    (ds/set-dataset-name "numeric-ds")))
(println "numeric-ds" (ds/shape numeric-ds))
(println (ds/head numeric-ds))


(println "Loading the trained-model")

(def model (pickle/load (builtins/open "model.pickle" "rb")))
(println "model" model)

(println "Loading the label->numeric lookup-table")

(def lookup-table (->  "lookup-table.edn"
                       slurp
                       edn/read-string
                       set/map-invert))
(println "lookup-table" lookup-table)
(def labels #_"ordered" (map lookup-table 
                             (-> lookup-table
                                 keys
                                 sort)))

;; for insiration look at Carsetn's code https://github.com/scicloj/sklearn-clj/blob/main/src/scicloj/sklearn_clj.clj

(println "Generating predictions using the model")

(def y_hat-py (py/py. model "predict" (-> numeric-ds
                                        dst/dataset->tensor
                                        py/->python)))

(def y_hat-ds (-> (ds/dataset {:species-predicted y_hat-py})
                  (ds/update-columns :species-predicted (fn [col] (map #(-> % int lookup-table) col)))
                  ))

(def y_hat-proba-py (py/py. model "predict_proba" (-> numeric-ds
                                                     dst/dataset->tensor
                                                     py/->python)))

(def proba-labels (map #(keyword (str % "-proba")) labels))
(def y_hat-proba-ds (->> y_hat-proba-py
                         (map #(zipmap proba-labels %))
                         ds/dataset)) 

(def prediction-ds (-> (ds/append to-be-predicted-ds y_hat-ds y_hat-proba-ds)
                       (ds/set-dataset-name "prediction-ds")))
(println "prediction-ds" (ds/shape prediction-ds))
(println (ds/head prediction-ds))

(println "Writing predictions to file")

(-> prediction-ds
    (ds/write-csv! "predictions.csv"))


(shutdown-agents)

