(require '[clojure.string :as str])

(def inp
  (partition
   2
   (map #(str/split % #" ")
        (str/split
         (slurp "../input/day8") #"\n| \| "))))


(defn pattern-pass1 [sol pattern]
  (let [c (count pattern)
        s (set pattern)]
    (cond (= c 2) (assoc sol 1 s)
          (= c 3) (assoc sol 7 s)
          (= c 4) (assoc sol 4 s)
          (= c 7) (assoc sol 8 s)
          : else sol)))
  
(defn pattern-pass2 [sol pattern]
  (let [c (count pattern)
        s (set pattern)]
    (cond (= c 5)
          (cond (= (count
                    (difference (sol 1) s))
                   1)
                (assoc sol 6 s)
                (= (count
                    (difference (sol 4) s))
                   0)
                (assoc sol 9 s)
                :else (assoc sol 0 s))
          (= c 6)
          (cond (= (count
                    (difference (sol 1) s))
                   0)
                (assoc sol 3 s)
                (= (count
                    (union (sol 4) s))
                   7)
                (assoc sol 2 s)
                :else (assoc sol 5 s))
          :else sol)))



(println (first inp))
  
