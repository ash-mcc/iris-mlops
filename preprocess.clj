(ns preprocess
  (:require [scicloj.ml.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.libs.arrow :as arrow]))

;; see  https://morioh.com/p/80f58933a6ab


(def raw-ds (-> "iris.csv"
                (ds/dataset {:key-fn keyword})))
(println "raw-ds" (ds/shape raw-ds))


(def numeric-ds (-> raw-ds
                    (ds/categorical->number cf/categorical)
                    (ds/set-dataset-name "numeric-ds")))
(println "numeric-ds" (ds/shape numeric-ds))


;; will need the :categorical-map from 
;; (meta (numeric-ds :species))


(def split (-> numeric-ds
               ds/train-test-split))
(printf "train-ds %s, test-ds %s" (-> split :train-ds ds/shape) (-> split :test-ds ds/shape))


(arrow/dataset->stream! (:train-ds split) "train.arrow")
(arrow/dataset->stream! (:test-ds split) "test.arrow")


(shutdown-agents)
;; (System/exit 0)