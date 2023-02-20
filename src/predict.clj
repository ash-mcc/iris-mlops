(ns predict
  (:require [clojure.data.json :as json]
            [scicloj.ml.dataset :as ds]
            [tech.v3.dataset.tensor :as dst]
            [libpython-clj2.python :as py]
            [libpython-clj2.require :refer [require-python]]))

(require-python '[builtins]
                '[pickle])


;; ------------------------------------------
;; Main fn
;;

(defn -main
  [& [model-filepath label-lookup-filepath predictions-filepath]]

  ;; specify the to-be-predicted data
  (let [col-names          [:sepal_length :sepal_width :petal_length :petal_width :species-truth]
        rows               [[5.1 3.5 1.4 0.2 "setosa"]
                            [7.0 3.2 4.7 1.4 "versicolor"]
                            [6.3 3.3 6.0 2.5 "virginica"]]
        seq-of-pairs       (->> col-names
                                (map-indexed vector)
                                (map (fn [[idx col-name]]
                                       [col-name (map #(nth % idx) rows)])))
        to-be-predicted-ds (ds/dataset seq-of-pairs)]
    (println "to-be-predicted-ds, rows x cols" (ds/shape to-be-predicted-ds))
    (println (ds/head to-be-predicted-ds))

    ;; drop the column that is to be predicted/estimated
    (let [numeric-ds (-> to-be-predicted-ds
                         (ds/drop-columns :species-truth)
                         (ds/set-dataset-name "numeric-ds"))]
      (println "numeric-ds, rows x cols" (ds/shape numeric-ds))
      (println (ds/head numeric-ds))


      ;; load the trained model
      (let [model (pickle/load (builtins/open model-filepath "rb"))]
        (println "Read" model-filepath)

        ;; load the numeric->categorical lookup table
        (let [label-lookup (-> (slurp label-lookup-filepath) 
                               (json/read-str :key-fn #(Integer/parseInt %)))]
          (println "Read label-lookup" label-lookup) 

          (let [;; generate estimates from the model
                y-hat          (py/py. model "predict" (-> numeric-ds
                                                           dst/dataset->tensor
                                                           py/->python))
                y-hat-ds       (-> (ds/dataset {:species-predicted y-hat})
                                   ;; map numerics -> category levels
                                   (ds/update-columns :species-predicted (fn [col]
                                                                           (map #(-> % int label-lookup)
                                                                                col))))

                ;; generate likelihoods from the model 
                y-hat-proba    (py/py. model "predict_proba" (-> numeric-ds
                                                                 dst/dataset->tensor
                                                                 py/->python))
                proba-labels   (->> (sort-by key label-lookup)
                                    (map second)
                                    (map #(keyword (str % "-proba"))))
                y-hat-proba-ds (->> y-hat-proba
                                    (map #(zipmap proba-labels %))
                                    ds/dataset)

                ;; combine into a single 'predictions' dataset
                prediction-ds  (-> (ds/append to-be-predicted-ds y-hat-ds y-hat-proba-ds)
                                   (ds/set-dataset-name "prediction-ds"))]

            ;; save the predications dataset
            (ds/write-csv! prediction-ds predictions-filepath)
            (println "Wrote" predictions-filepath ", rows x cols" (ds/shape prediction-ds))
            (println (ds/head prediction-ds))))))))


;; ------------------------------------------
;; For being run as a script 
;;

(apply -main *command-line-args*)
(shutdown-agents)


;; ------------------------------------------
;; Jotter 
;;
;;

(comment
  
  (-main "model.pkl" "data/label-lookup.json" "data/predictions.csv")
  
  )