(ns pretrain
  (:require [scicloj.ml.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.libs.arrow :as arrow]))


(println "Loading the dataset for training & testing")

(def raw-ds (-> "iris.csv"
                (ds/dataset {:key-fn keyword})))
(println "iris.csv" (ds/shape raw-ds))
(println (ds/head raw-ds))


(println "Featurising the dataset")

;; see  https://morioh.com/p/80f58933a6ab

(def numeric-ds (-> raw-ds
                    (ds/categorical->number cf/categorical)
                    (ds/set-dataset-name "numeric-ds")))
(println "numeric-ds" (ds/shape numeric-ds))
(println (ds/head numeric-ds))


(println "Splitting dataset for training & testing")

(def split (-> numeric-ds
               ds/train-test-split))
(printf "train-ds %s, test-ds %s" (-> split :train-ds ds/shape) (-> split :test-ds ds/shape))


(println "Writing label->numeric lookup-table to file")

(spit "lookup-table.edn" (-> numeric-ds
                             :species
                             meta
                             :categorical-map
                             :lookup-table
                             pr-str))

(println "Writing training & testing dataset to files")

(arrow/dataset->stream! (:train-ds split) "train.arrow")
(arrow/dataset->stream! (:test-ds split) "test.arrow")


(shutdown-agents)
