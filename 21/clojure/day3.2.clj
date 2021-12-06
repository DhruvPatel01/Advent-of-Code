(def a-seq (sort (clojure.string/split (slurp *in*) #"\n")))

(defn to-bin [a] (mapv #(Character/getNumericValue %) a))

(def a-seq (mapv to-bin a-seq))
(defn from-bin [a]
  (loop [acc 0
         [head & rest] a]
    (if (nil? head) acc (recur (+ (* 2 acc) head) rest))))

(defn first-one [s e i]
  (if (= s e) s 
      (let [m (quot (+ s e) 2)]
        (if  (= 1 ((a-seq m) i))
          (recur s m i)
          (recur (+ m 1) e i)))))

(defn zero-minor? [s e j]
  (< (- j 1) (quot (+ s e) 2)))
(defn one-major? [s e j]
  (<= j (quot (+ s e) 2)))


(defn part2 [s e i t]
  (if (= s (- e t)) s
      (let [j (first-one s e i)]
        (cond (= j e) (recur s e (+ i 1) t)
              (= t 0) (if (zero-minor? s e j)
                        (recur s j (+ i 1) t)
                        (recur j e (+ i 1) t))
              (= t 1) (if (one-major? s e j)
                        (recur j e (+ i 1) t)
                        (recur s j (+ i 1) t))
          )
        )))



(let [o2  (part2 0 (count a-seq) 0 0) 
      co2 (part2 0 (count a-seq) 0 1)
      o2 (from-bin (a-seq o2))
      co2 (from-bin (a-seq co2))]

  (println (* o2 co2)))




