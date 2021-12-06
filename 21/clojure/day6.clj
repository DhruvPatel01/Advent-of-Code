(require '[clojure.string :as str])
(declare g)
(declare h)

(def g (memoize
        (fn f [n m]
          (if (< n (+ m 9)) 1
              (+ (g n (+ m 9)) (h n (+ m 16)))))))

(def h (memoize (fn  [n m]
                  (if (< n m) 1
                      (+ (g n m) (h n (+ m 7)))))))


(def x (map #(Integer/parseInt %)
            (str/split (str/trim (slurp *in*)) #",")))

(println (str "Part1 "
              (apply + (map #(h 80 (+ % 1)) x))))

(println (str "Part2 "
              (apply + (map #(h 256 (+ % 1)) x))))
