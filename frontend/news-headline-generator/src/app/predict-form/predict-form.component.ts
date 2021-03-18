import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { PredictService } from '../services/predict.service';

@Component({
  selector: 'app-predict-form',
  templateUrl: './predict-form.component.html',
  styleUrls: ['./predict-form.component.scss'],
})
export class PredictFormComponent implements OnInit {
  input = '';
  output = '-';

  get getInput(): string {
    return this.input;
  }

  get getOutput(): string {
    return this.output;
  }

  constructor(private _predictService: PredictService) {}

  onSubmit() {
    this._predictService.predict(this.input).subscribe(
      (data) => {
        console.log('Success!');
        console.log(data);
        this.output = data;
      },
      (error) => {
        console.log('Error occured!');
      }
    );
  }

  ngOnInit(): void {}
}
