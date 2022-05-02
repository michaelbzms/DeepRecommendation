import {Component} from '@angular/core';
import {BackendService} from "./services/backend.service";
import {Movie} from "./models/movies";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'ui';
  movies$ = this.backendService.getMovies();
  movies: Movie[] = []

  constructor(private backendService: BackendService) {
    this.movies$.subscribe(dict => {
      this.movies = [];
      for (const [imdbID, data] of Object.entries(dict)) {
        // @ts-ignore
        this.movies.push(new Movie(imdbID, data['primaryTitle'], data['startYear']));
      }
      console.log(this.movies);
    })
  }

}
