(ns prep
  (:require [clojure.set :as set]
            [clojure.data.json :as json]
            [scicloj.ml.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.libs.arrow :as arrow]))


;; ------------------------------------------
;; Main fn
;;

(defn -main
  [& [iris-filepath label-lookup-filepath train-filepath test-filepath]]

  ;; load the raw source data
  (let [raw-ds (ds/dataset iris-filepath {:key-fn keyword})]
    (println "Read" iris-filepath ", rows x cols" (ds/shape raw-ds))
    (println (ds/head raw-ds))
    
    ;; categorical->numeric values
    (let [numeric-ds (-> raw-ds
                         (ds/categorical->number cf/categorical)
                         (ds/set-dataset-name "numeric-ds"))]
      (println "Categorically encoded dataset, rows x cols" (ds/shape numeric-ds))
      (println (ds/head numeric-ds))

      ;; save the numeric->categorical lookup table
      (let [label-lookup (-> numeric-ds
                             :species
                             meta
                             :categorical-map
                             :lookup-table
                             set/map-invert)]
        (spit label-lookup-filepath (json/write-str label-lookup))
        (println "Wrote" label-lookup-filepath ", count" (count label-lookup))

        ;; save the train/test data
        (let [split (ds/train-test-split numeric-ds)]
          (arrow/dataset->stream! (:train-ds split) train-filepath)
          (println "Wrote" train-filepath ", rows x cols" (ds/shape (:train-ds split)))
          (arrow/dataset->stream! (:test-ds split) test-filepath)
          (println "Wrote" test-filepath ", rows x cols" (ds/shape (:test-ds split))))))))


;; ------------------------------------------
;; For being run as a script 
;;

(apply -main *command-line-args*)
(shutdown-agents)