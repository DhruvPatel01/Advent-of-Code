(def a-seq (clojure.string/split (slurp *in*) #"\n"))
(defn to-bin [a]
  (map #(Character/getNumericValue %) a))

(def seq-len (count a-seq))
(def a-seq (map to-bin a-seq))

(defn reducer [a b] (map #(+ %1  %2) a b))

(def part1-partial (reduce reducer a-seq))

(defn gamma-bin [a]
  (map #(if (> (* 2 %) seq-len) 1 0) a))

(defn epsi-bin [gamma] (map #(if (= % 0) 1 0) gamma))

(let [gamma (gamma-bin part1-partial)
      epsil (epsi-bin gamma)
      gamma (apply str gamma)
      epsil (apply str epsil)
      gamma (Integer/parseInt gamma 2)
      epsil (Integer/parseInt epsil 2)]

  (println (* gamma epsil)))


